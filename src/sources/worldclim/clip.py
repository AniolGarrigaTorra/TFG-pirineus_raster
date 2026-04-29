from pathlib import Path
import zipfile

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

from src.io.paths import ensure_dir, get_source_clipped_dir
from src.sources.worldclim.naming import (
    build_worldclim_zip_path,
    build_worldclim_clipped_name,
    build_worldclim_monthly_member_basename,
    build_worldclim_static_index_member_basename,
    build_worldclim_static_single_member_basename,
    get_layer_structure,
    get_source_resolution,
    get_zip_variable_codes,
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

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    return transformer.transform_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        densify_pts=21,
    )


def _find_tif_in_zip(zip_path: Path, expected_basename: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    matches = [
        member
        for member in members
        if Path(member).name == expected_basename
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        sample = "\n".join(members[:20])
        raise FileNotFoundError(
            f"Could not find {expected_basename} inside {zip_path}.\n"
            f"First ZIP members:\n{sample}"
        )

    raise RuntimeError(
        f"Found multiple matches for {expected_basename} inside {zip_path}: {matches}"
    )


def _open_zip_member_path(zip_path: Path, member: str) -> str:
    return f"/vsizip/{zip_path.resolve()}/{member}"


def clip_one_raster(
    zip_path: Path,
    member: str,
    output_path: Path,
    clip_bounds_in_source_crs: tuple[float, float, float, float],
    compression: str = "LZW",
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[clip] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)

    raster_path = _open_zip_member_path(zip_path, member)

    with rasterio.open(raster_path) as src:
        window = from_bounds(
            *clip_bounds_in_source_crs,
            transform=src.transform,
        )

        window = window.round_offsets().round_lengths()

        data = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "compress": compression,
            }
        )

        for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
            profile.pop(key, None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

            dst.update_tags(
                SOURCE_CLIPPED_FROM=str(zip_path.name),
                SOURCE_MEMBER=member,
                CLIP_METHOD="bbox",
                CLIP_BOUNDS_SOURCE_CRS=str(clip_bounds_in_source_crs),
                NOTE=(
                    "Intermediate clipped raster. Original WorldClim values are preserved; "
                    "scale_factor is not applied here."
                ),
            )

    print(f"[clip] Written: {output_path}")


def _get_enabled_monthly_variables(source_cfg: dict) -> list[str]:
    variables_cfg = source_cfg.get("variables", {})
    enabled = [
        variable
        for variable, cfg in variables_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled monthly variables found in source config.")

    return enabled


def _get_enabled_indices(source_cfg: dict) -> list[tuple[str, int]]:
    indices_cfg = source_cfg.get("indices", {})
    enabled = []

    for index_name, cfg in indices_cfg.items():
        if cfg.get("enabled", False):
            enabled.append((index_name, int(cfg["index"])))

    if not enabled:
        raise ValueError("No enabled indices found in source config.")

    return enabled


def _get_enabled_static_variables(source_cfg: dict) -> list[str]:
    variables_cfg = source_cfg.get("variables", {})
    enabled = [
        variable
        for variable, cfg in variables_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled static variables found in source config.")

    return enabled


def _delete_zip_if_safe(
    zip_path: Path,
    keep_global_zip_after_clip: bool,
) -> None:
    if keep_global_zip_after_clip:
        return

    if zip_path.exists():
        print(f"[clip] Removing global raw ZIP after clipping: {zip_path}")
        zip_path.unlink()


def clip_monthly_climatology(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str,
    overwrite: bool,
    keep_global_zip_after_clip: bool,
) -> list[Path]:
    source = source_cfg["source"]
    provider = source["provider"]
    product = source["product"]
    source_resolution = get_source_resolution(source_cfg)

    raw_dir = (
        Path(project_cfg["paths"]["raw_dir"])
        / provider
        / product
        / source_resolution
    )

    written_paths: list[Path] = []

    for variable in _get_enabled_monthly_variables(source_cfg):
        zip_path = build_worldclim_zip_path(
            raw_dir=raw_dir,
            source_cfg=source_cfg,
            zip_variable_code=variable,
        )

        if not zip_path.exists():
            raise FileNotFoundError(f"Missing raw WorldClim ZIP: {zip_path}")

        print(f"[clip] Processing monthly variable: {variable}")
        print(f"[clip] ZIP: {zip_path}")

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=variable,
        )

        ensure_dir(clipped_dir)

        for month in range(1, 13):
            expected_basename = build_worldclim_monthly_member_basename(
                source_cfg=source_cfg,
                variable=variable,
                month=month,
            )

            member = _find_tif_in_zip(zip_path, expected_basename)

            output_name = build_worldclim_clipped_name(
                source_cfg=source_cfg,
                layer_name=variable,
                domain_name=clip_aoi_name,
                month=month,
            )

            output_path = clipped_dir / output_name

            clip_one_raster(
                zip_path=zip_path,
                member=member,
                output_path=output_path,
                clip_bounds_in_source_crs=clip_bounds_source_crs,
                compression=compression,
                overwrite=overwrite,
            )

            written_paths.append(output_path)

        _delete_zip_if_safe(zip_path, keep_global_zip_after_clip)

    return written_paths


def clip_static_index_set(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str,
    overwrite: bool,
    keep_global_zip_after_clip: bool,
) -> list[Path]:
    source = source_cfg["source"]
    provider = source["provider"]
    product = source["product"]
    source_resolution = get_source_resolution(source_cfg)
    zip_variable_code = source_cfg["dataset"]["zip_variable_code"]

    raw_dir = (
        Path(project_cfg["paths"]["raw_dir"])
        / provider
        / product
        / source_resolution
    )

    zip_path = build_worldclim_zip_path(
        raw_dir=raw_dir,
        source_cfg=source_cfg,
        zip_variable_code=zip_variable_code,
    )

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing raw WorldClim ZIP: {zip_path}")

    written_paths: list[Path] = []

    for index_name, index_number in _get_enabled_indices(source_cfg):
        print(f"[clip] Processing static index: {index_name}")

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=index_name,
        )

        ensure_dir(clipped_dir)

        expected_basename = build_worldclim_static_index_member_basename(
            source_cfg=source_cfg,
            index_number=index_number,
        )

        member = _find_tif_in_zip(zip_path, expected_basename)

        output_name = build_worldclim_clipped_name(
            source_cfg=source_cfg,
            layer_name=index_name,
            domain_name=clip_aoi_name,
        )

        output_path = clipped_dir / output_name

        clip_one_raster(
            zip_path=zip_path,
            member=member,
            output_path=output_path,
            clip_bounds_in_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
        )

        written_paths.append(output_path)

    _delete_zip_if_safe(zip_path, keep_global_zip_after_clip)

    return written_paths


def clip_static_single(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str,
    overwrite: bool,
    keep_global_zip_after_clip: bool,
) -> list[Path]:
    source = source_cfg["source"]
    provider = source["provider"]
    product = source["product"]
    source_resolution = get_source_resolution(source_cfg)
    zip_variable_code = source_cfg["dataset"]["zip_variable_code"]

    raw_dir = (
        Path(project_cfg["paths"]["raw_dir"])
        / provider
        / product
        / source_resolution
    )

    zip_path = build_worldclim_zip_path(
        raw_dir=raw_dir,
        source_cfg=source_cfg,
        zip_variable_code=zip_variable_code,
    )

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing raw WorldClim ZIP: {zip_path}")

    written_paths: list[Path] = []

    for variable in _get_enabled_static_variables(source_cfg):
        print(f"[clip] Processing static variable: {variable}")

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=variable,
        )

        ensure_dir(clipped_dir)

        expected_basename = build_worldclim_static_single_member_basename(
            source_cfg=source_cfg,
            variable=variable,
        )

        member = _find_tif_in_zip(zip_path, expected_basename)

        output_name = build_worldclim_clipped_name(
            source_cfg=source_cfg,
            layer_name=variable,
            domain_name=clip_aoi_name,
        )

        output_path = clipped_dir / output_name

        clip_one_raster(
            zip_path=zip_path,
            member=member,
            output_path=output_path,
            clip_bounds_in_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
        )

        written_paths.append(output_path)

    _delete_zip_if_safe(zip_path, keep_global_zip_after_clip)

    return written_paths


def clip_worldclim_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    download_cfg = source_cfg.get("download", {})
    output_cfg = source_cfg.get("output", {})

    source_crs = source.get("source_crs", "EPSG:4326")
    layer_structure = get_layer_structure(source_cfg)

    clip_aoi_name = clip_aoi_cfg["name"]
    clip_aoi_crs = clip_aoi_cfg["crs"]

    compression = output_cfg.get("compression", "LZW")
    overwrite = bool(download_cfg.get("overwrite_existing", False))
    keep_global_zip_after_clip = bool(
        download_cfg.get("keep_global_zip_after_clip", True)
    )

    clip_bounds_project_crs = _get_aoi_bounds(clip_aoi_cfg)

    clip_bounds_source_crs = _transform_bounds(
        bounds=clip_bounds_project_crs,
        src_crs=clip_aoi_crs,
        dst_crs=source_crs,
    )

    print("[clip] AOI:", clip_aoi_name)
    print("[clip] AOI CRS:", clip_aoi_crs)
    print("[clip] Source CRS:", source_crs)
    print("[clip] Layer structure:", layer_structure)
    print("[clip] Bounds in AOI CRS:", clip_bounds_project_crs)
    print("[clip] Bounds in source CRS:", clip_bounds_source_crs)

    if layer_structure == "monthly_climatology":
        return clip_monthly_climatology(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            clip_bounds_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
            keep_global_zip_after_clip=keep_global_zip_after_clip,
        )

    if layer_structure == "static_index_set":
        return clip_static_index_set(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            clip_bounds_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
            keep_global_zip_after_clip=keep_global_zip_after_clip,
        )

    if layer_structure == "static_single":
        return clip_static_single(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            clip_bounds_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite,
            keep_global_zip_after_clip=keep_global_zip_after_clip,
        )

    raise NotImplementedError(f"Unsupported layer_structure: {layer_structure}")