from __future__ import annotations

from pathlib import Path
import re
import zipfile

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

from src.io.paths import ensure_dir, get_source_raw_dir, get_source_clipped_dir
from src.pipeline.progress import progress_log
from src.sources.copernicus.naming import (
    validate_copernicus_source_config,
    get_enabled_variable_items,
    get_file_format,
    get_source_resolution,
    get_file_spec_for_variable,
    build_copernicus_raw_path,
    build_copernicus_clipped_name,
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
    xmin, ymin, xmax, ymax = bounds

    if str(src_crs) == str(dst_crs):
        return xmin, ymin, xmax, ymax

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    return transformer.transform_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        densify_pts=21,
    )


def _find_tif_in_zip(
    zip_path: Path,
    zip_member: str | None = None,
    zip_member_pattern: str | None = None,
) -> str:
    """
    Find a GeoTIFF member inside a ZIP.

    Priority:
      1. exact zip_member
      2. regex zip_member_pattern
      3. if only one .tif/.tiff exists, use it
    """
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

        if len(matches) == 0:
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

        if len(matches) == 0:
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
        f"Could not choose a TIFF inside {zip_path} because {len(tif_members)} "
        f"TIFF files were found.\n"
        "Set download.files.<variable>.zip_member or zip_member_pattern in YAML.\n"
        f"First TIFF members:\n{sample}"
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

        # rasterio/GDAL virtual path for ZIP members
        return f"zip://{raw_path}!{member}"

    raise NotImplementedError(f"Unsupported file_format={file_format!r}")


def _get_clipped_output_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
) -> Path:
    source = source_cfg["source"]
    source_resolution = get_source_resolution(source_cfg)

    clipped_dir = get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=clip_aoi_name,
        source_resolution=source_resolution,
        variable=variable,
    )

    clipped_name = build_copernicus_clipped_name(
        source_cfg=source_cfg,
        variable=variable,
        domain_name=clip_aoi_name,
    )

    return clipped_dir / clipped_name


def _clip_one_raster(
    input_raster_path: str,
    output_path: Path,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str = "LZW",
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        progress_log(f"[clip] Cache hit: {output_path}")
        return output_path

    ensure_dir(output_path.parent)

    with rasterio.open(input_raster_path) as src:
        window = from_bounds(
            *clip_bounds_source_crs,
            transform=src.transform,
        )

        window = window.round_offsets().round_lengths()

        data = src.read(window=window)

        transform = src.window_transform(window)
        profile = src.profile.copy()

        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress=compression,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
            dst.update_tags(
                clipped_from=input_raster_path,
                clip_bounds_source_crs=str(clip_bounds_source_crs),
            )

    progress_log(f"[clip] Written: {output_path}")
    return output_path


def clip_copernicus_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    """
    Clip Copernicus raw rasters to the clipping AOI.

    This is generic for static Copernicus raster products.
    """
    validate_copernicus_source_config(source_cfg)

    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {}) or {}
    download_cfg = source_cfg.get("download", {}) or {}

    provider = source["provider"]
    product = source["product"]
    source_resolution = str(processing["source_resolution"])

    clip_aoi_name = clip_aoi_cfg["name"]
    clip_aoi_crs = clip_aoi_cfg["crs"]
    clip_bounds_aoi_crs = _get_aoi_bounds(clip_aoi_cfg)

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        source_resolution=source_resolution,
    )

    compression = str(output_cfg.get("compression", "LZW"))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    progress_log(f"[clip] Provider: {provider}")
    progress_log(f"[clip] Product: {product}")
    progress_log(f"[clip] AOI: {clip_aoi_name}")
    progress_log(f"[clip] AOI CRS: {clip_aoi_crs}")
    progress_log(f"[clip] Raw dir: {raw_dir}")
    progress_log(f"[clip] File format: {get_file_format(source_cfg)}")

    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        raw_path = build_copernicus_raw_path(
            raw_dir=raw_dir,
            source_cfg=source_cfg,
            variable=variable,
        )

        if not raw_path.exists():
            # If file doesn't exist, treat as optional
            # (this can happen when variables are filtered during download)
            progress_log(
                f"[clip] Raw file not found for variable={variable}. "
                f"Skipping (likely filtered during download): {raw_path}"
            )
            continue

        input_raster_path = _open_raster_path_for_variable(
            raw_path=raw_path,
            source_cfg=source_cfg,
            variable=variable,
        )

        with rasterio.open(input_raster_path) as src:
            source_crs = src.crs

            if source_crs is None:
                configured_source_crs = source.get("source_crs")
                if configured_source_crs is None:
                    raise ValueError(
                        f"Raster has no CRS and source.source_crs is not set: "
                        f"{input_raster_path}"
                    )
                source_crs = configured_source_crs

            clip_bounds_source_crs = _transform_bounds(
                bounds=clip_bounds_aoi_crs,
                src_crs=clip_aoi_crs,
                dst_crs=str(source_crs),
            )

        output_path = _get_clipped_output_path(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            variable=variable,
        )

        progress_log(f"[clip] Variable: {variable}")
        progress_log(f"[clip] Description: {variable_cfg.get('description', '')}")
        progress_log(f"[clip] Raw path: {raw_path}")
        progress_log(f"[clip] Raster path: {input_raster_path}")
        progress_log(f"[clip] Source CRS: {source_crs}")
        progress_log(f"[clip] Bounds in AOI CRS: {clip_bounds_aoi_crs}")
        progress_log(f"[clip] Bounds in source CRS: {clip_bounds_source_crs}")
        progress_log(f"[clip] Output: {output_path}")

        written_path = _clip_one_raster(
            input_raster_path=input_raster_path,
            output_path=output_path,
            clip_bounds_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
        )

        written_paths.append(written_path)

    return written_paths
