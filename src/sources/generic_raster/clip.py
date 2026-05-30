from __future__ import annotations

from pathlib import Path
import re
import zipfile

import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from src.io.paths import ensure_dir, get_source_clipped_dir, get_source_raw_dir
from src.sources.generic_raster.naming import (
    build_clipped_name,
    build_raw_path,
    get_enabled_variable_items,
    get_file_format,
    get_file_spec_for_variable,
    get_source_resolution,
    validate_generic_raster_source_config,
)


def _get_aoi_bounds(aoi_cfg: dict) -> tuple[float, float, float, float]:
    bounds = aoi_cfg["bounds"]
    return (
        float(bounds["xmin"]),
        float(bounds["ymin"]),
        float(bounds["xmax"]),
        float(bounds["ymax"]),
    )


def _transform_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    if str(src_crs) == str(dst_crs):
        return bounds
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform_bounds(*bounds, densify_pts=21)


def _find_tif_in_zip(
    zip_path: Path,
    zip_member: str | None = None,
    zip_member_pattern: str | None = None,
) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    tif_members = [
        member
        for member in members
        if member.lower().endswith((".tif", ".tiff"))
    ]

    if zip_member:
        matches = [
            member
            for member in members
            if member == zip_member or Path(member).name == zip_member
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Could not find zip_member={zip_member!r} inside {zip_path}"
            )
        raise RuntimeError(
            f"Multiple matches for zip_member={zip_member!r} inside {zip_path}: "
            f"{matches}"
        )

    if zip_member_pattern:
        pattern = re.compile(zip_member_pattern)
        matches = [
            member
            for member in tif_members
            if pattern.search(member) or pattern.search(Path(member).name)
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            sample = "\n".join(tif_members[:30])
            raise FileNotFoundError(
                f"No TIFF in {zip_path} matched pattern={zip_member_pattern!r}.\n"
                f"First TIFF members:\n{sample}"
            )
        raise RuntimeError(
            f"Multiple TIFFs in {zip_path} matched pattern={zip_member_pattern!r}: "
            f"{matches[:20]}"
        )

    if len(tif_members) == 1:
        return tif_members[0]

    sample = "\n".join(tif_members[:30])
    raise RuntimeError(
        f"Could not choose a TIFF inside {zip_path}; found {len(tif_members)} "
        f"TIFF files.\nFirst TIFF members:\n{sample}"
    )


def _open_raster_path_for_variable(
    raw_path: Path,
    source_cfg: dict,
    variable: str,
) -> str:
    file_format = get_file_format(source_cfg)
    if file_format == "geotiff":
        return str(raw_path)
    if file_format == "zip_geotiff":
        spec = get_file_spec_for_variable(source_cfg, variable)
        member = _find_tif_in_zip(
            zip_path=raw_path,
            zip_member=spec.get("zip_member"),
            zip_member_pattern=spec.get("zip_member_pattern"),
        )
        return f"zip://{raw_path}!{member}"
    raise NotImplementedError(f"Unsupported file_format={file_format!r}")


def _clip_one_raster(
    input_raster_path: str,
    output_path: Path,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str = "LZW",
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"[clip] Exists, skipping: {output_path}")
        return output_path

    ensure_dir(output_path.parent)

    with rasterio.open(input_raster_path) as src:
        window = from_bounds(*clip_bounds_source_crs, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress=compression,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            dst.update_tags(
                clipped_from=input_raster_path,
                clip_bounds_source_crs=str(clip_bounds_source_crs),
            )

    print(f"[clip] Written: {output_path}")
    return output_path


def clip_generic_raster_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    provider: str | None = None,
) -> list[Path]:
    validate_generic_raster_source_config(source_cfg, provider=provider)

    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {}) or {}
    download_cfg = source_cfg.get("download", {}) or {}

    provider_name = source["provider"]
    product = source["product"]
    source_resolution = str(processing["source_resolution"])
    clip_aoi_name = clip_aoi_cfg["name"]
    clip_bounds_aoi_crs = _get_aoi_bounds(clip_aoi_cfg)

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=provider_name,
        product=product,
        source_resolution=source_resolution,
    )
    compression = str(output_cfg.get("compression", "LZW"))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    print("[clip] Provider:", provider_name)
    print("[clip] Product:", product)
    print("[clip] AOI:", clip_aoi_name)
    print("[clip] Raw dir:", raw_dir)

    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        raw_path = build_raw_path(raw_dir, source_cfg, variable)
        if not raw_path.exists():
            if not bool(variable_cfg.get("required", True)):
                print(f"[clip] Optional raw raster missing, skipping: {raw_path}")
                continue
            raise FileNotFoundError(f"Missing raw raster for variable={variable}: {raw_path}")

        input_raster_path = _open_raster_path_for_variable(raw_path, source_cfg, variable)
        with rasterio.open(input_raster_path) as src:
            source_crs = src.crs or source.get("source_crs")
            if source_crs is None:
                raise ValueError(
                    f"Raster has no CRS and source.source_crs is not set: {input_raster_path}"
                )
            clip_bounds_source_crs = _transform_bounds(
                bounds=clip_bounds_aoi_crs,
                src_crs=str(clip_aoi_cfg["crs"]),
                dst_crs=str(source_crs),
            )

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider_name,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=get_source_resolution(source_cfg),
            variable=variable,
        )
        output_path = clipped_dir / build_clipped_name(source_cfg, variable, clip_aoi_name)

        print("==============================")
        print(f"[clip] Variable: {variable}")
        print(f"[clip] Description: {variable_cfg.get('description', '')}")
        print(f"[clip] Raw path: {raw_path}")
        print(f"[clip] Raster path: {input_raster_path}")
        print(f"[clip] Output: {output_path}")

        written_paths.append(
            _clip_one_raster(
                input_raster_path=input_raster_path,
                output_path=output_path,
                clip_bounds_source_crs=clip_bounds_source_crs,
                compression=compression,
                overwrite=overwrite,
            )
        )

    return written_paths
