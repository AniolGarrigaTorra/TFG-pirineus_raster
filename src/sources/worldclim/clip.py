from pathlib import Path
import zipfile

import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer

from src.io.paths import ensure_dir, get_source_clipped_dir
from src.sources.worldclim.download import get_enabled_variables
from src.sources.worldclim.naming import (
    build_worldclim_zip_path,
    build_worldclim_clipped_month_name,
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

    # transform_bounds handles bbox curvature better than transforming only two corners
    return transformer.transform_bounds(
        xmin,
        ymin,
        xmax,
        ymax,
        densify_pts=21,
    )


def _find_monthly_tif_in_zip(
    zip_path: Path,
    source_resolution: str,
    variable: str,
    month: int,
) -> str:
    """
    Find the TIFF corresponding to one WorldClim variable and month inside the ZIP.

    Expected examples:
      wc2.1_10m_tmin_01.tif
      wc2.1_30s_prec_12.tif

    The TIFF may be inside a folder in the ZIP, so we compare basenames.
    """
    expected_basename = f"wc2.1_{source_resolution}_{variable}_{month:02d}.tif"

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
        raise FileNotFoundError(
            f"Could not find {expected_basename} inside {zip_path}"
        )

    raise RuntimeError(
        f"Found multiple matches for {expected_basename} inside {zip_path}: {matches}"
    )


def _open_zip_member_path(zip_path: Path, member: str) -> str:
    """
    Build a GDAL /vsizip/ path readable by rasterio.
    """
    return f"/vsizip/{zip_path.resolve()}/{member}"


def clip_one_month(
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

        # Round window to full pixels
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

        # The source profile may contain tiling/block parameters.
        # After clipping, these can become invalid if block sizes are not multiples of 16
        # or larger than the clipped raster dimensions.
        for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
            profile.pop(key, None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

            dst.update_tags(
                SOURCE_CLIPPED_FROM=str(zip_path.name),
                SOURCE_MEMBER=member,
                CLIP_METHOD="bbox",
                CLIP_BOUNDS_SOURCE_CRS=str(clip_bounds_in_source_crs),
                NOTE="Intermediate clipped raster. Original WorldClim values are preserved; scale_factor is not applied here.",
            )

    print(f"[clip] Written: {output_path}")


def clip_worldclim_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    download_cfg = source_cfg.get("download", {})
    output_cfg = source_cfg.get("output", {})

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]
    source_crs = source.get("source_crs", "EPSG:4326")

    clip_aoi_name = clip_aoi_cfg["name"]
    clip_aoi_crs = clip_aoi_cfg["crs"]

    compression = output_cfg.get("compression", "LZW")
    overwrite = bool(download_cfg.get("overwrite_existing", False))
    keep_global_zip_after_clip = bool(
        download_cfg.get("keep_global_zip_after_clip", True)
    )

    raw_dir = (
        Path(project_cfg["paths"]["raw_dir"])
        / provider
        / product
        / source_resolution
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
    print("[clip] Bounds in AOI CRS:", clip_bounds_project_crs)
    print("[clip] Bounds in source CRS:", clip_bounds_source_crs)

    enabled_variables = get_enabled_variables(source_cfg)
    written_paths: list[Path] = []

    for variable in enabled_variables:
        zip_path = build_worldclim_zip_path(
            raw_dir=raw_dir,
            source_resolution=source_resolution,
            variable=variable,
        )

        if not zip_path.exists():
            raise FileNotFoundError(
                f"Missing raw WorldClim ZIP for variable '{variable}': {zip_path}"
            )

        print(f"[clip] Processing variable: {variable}")
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
            member = _find_monthly_tif_in_zip(
                zip_path=zip_path,
                source_resolution=source_resolution,
                variable=variable,
                month=month,
            )

            output_name = build_worldclim_clipped_month_name(
                source_resolution=source_resolution,
                variable=variable,
                month=month,
                domain_name=clip_aoi_name,
            )

            output_path = clipped_dir / output_name

            clip_one_month(
                zip_path=zip_path,
                member=member,
                output_path=output_path,
                clip_bounds_in_source_crs=clip_bounds_source_crs,
                compression=compression,
                overwrite=overwrite,
            )

            written_paths.append(output_path)

        if not keep_global_zip_after_clip:
            print(f"[clip] Removing global raw ZIP after clipping: {zip_path}")
            zip_path.unlink()

    return written_paths