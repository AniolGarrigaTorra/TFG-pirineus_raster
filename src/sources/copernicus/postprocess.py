from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile
from typing import Any

import rasterio
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from src.io.config import load_yaml, resolve_path


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def copy_single_file(
    *,
    input_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[postprocess] Exists, skipping: {output_path}")
        return output_path

    if len(input_paths) != 1:
        listing = "\n".join(str(p) for p in input_paths[:50])
        raise RuntimeError(
            "copy_single requires exactly one input file.\n"
            f"Received {len(input_paths)} files:\n{listing}"
        )

    ensure_parent(output_path)

    if output_path.exists() and overwrite:
        output_path.unlink()

    print(f"[postprocess] Copy single file:")
    print(f"  from: {input_paths[0]}")
    print(f"  to:   {output_path}")

    shutil.copy2(input_paths[0], output_path)
    return output_path


def find_tif_members_in_zip(
    zip_path: Path,
    zip_member_pattern: str | None = None,
) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    tif_members = [
        member
        for member in members
        if member.lower().endswith((".tif", ".tiff"))
    ]

    if zip_member_pattern:
        pattern = re.compile(zip_member_pattern)
        tif_members = [
            member
            for member in tif_members
            if pattern.search(member) or pattern.search(Path(member).name)
        ]

    return sorted(tif_members)


def zip_member_to_rasterio_uri(zip_path: Path, member: str) -> str:
    return f"zip://{zip_path}!{member}"


def collect_mixed_raster_uris(
    *,
    input_paths: list[Path],
    zip_member_pattern: str | None,
    allow_multiple_zip_members: bool = False,
    skip_zip_without_matching_members: bool = False,
) -> tuple[list[str], int]:
    raster_uris: list[str] = []
    zip_count = 0

    for path in input_paths:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            raster_uris.append(str(path))
            continue

        if suffix != ".zip":
            continue

        zip_count += 1
        members = find_tif_members_in_zip(
            zip_path=path,
            zip_member_pattern=zip_member_pattern,
        )

        if len(members) == 0:
            message = (
                f"No GeoTIFF members found in {path} "
                f"with zip_member_pattern={zip_member_pattern!r}"
            )
            if skip_zip_without_matching_members:
                print(f"[postprocess] {message}. Skipping ZIP.")
                continue
            raise FileNotFoundError(message)

        if len(members) > 1 and not allow_multiple_zip_members:
            listing = "\n".join(members[:30])
            raise RuntimeError(
                f"More than one GeoTIFF member found in {path}.\n"
                "Please set a more restrictive zip_member_pattern, or set:\n"
                "  allow_multiple_zip_members: true\n"
                "if all matching TIFFs should be mosaicked.\n"
                f"Members:\n{listing}"
            )

        for member in members:
            raster_uris.append(zip_member_to_rasterio_uri(path, member))

    return raster_uris, zip_count


def _bounds_from_aoi_cfg(aoi_cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    bounds = aoi_cfg.get("bounds", {}) or {}
    return (
        float(bounds["xmin"]),
        float(bounds["ymin"]),
        float(bounds["xmax"]),
        float(bounds["ymax"]),
    )


def _configured_source_crs(source_cfg: dict[str, Any]) -> CRS | None:
    for section_name in ("source", "dataset", "processing"):
        value = (source_cfg.get(section_name, {}) or {}).get("source_crs")
        if value:
            return CRS.from_user_input(value)
    return None


def _clip_aoi_bounds_for_crs(
    *,
    source_cfg: dict[str, Any],
    target_crs: CRS,
) -> tuple[float, float, float, float]:
    domains = source_cfg.get("domains", {}) or {}
    clip_aoi_config = domains.get("clip_aoi_config")

    if not clip_aoi_config:
        raise ValueError(
            "postprocess=mosaic_mixed_geotiff_to_clip_aoi requires "
            "domains.clip_aoi_config in the source config."
        )

    aoi_path = resolve_path(
        clip_aoi_config,
        base_path=source_cfg.get("_config_path"),
        must_exist=True,
    )
    aoi_cfg = load_yaml(aoi_path)
    aoi_crs = CRS.from_user_input(aoi_cfg["crs"])
    bounds = _bounds_from_aoi_cfg(aoi_cfg)

    if aoi_crs == target_crs:
        return bounds

    return transform_bounds(
        aoi_crs,
        target_crs,
        *bounds,
        densify_pts=21,
    )


def _bounds_intersect(
    left_a: float,
    bottom_a: float,
    right_a: float,
    top_a: float,
    left_b: float,
    bottom_b: float,
    right_b: float,
    top_b: float,
) -> bool:
    return left_a < right_b and right_a > left_b and bottom_a < top_b and top_a > bottom_b


def _crs_equivalent(left: CRS, right: CRS) -> bool:
    if left == right:
        return True

    left_epsg = left.to_epsg()
    right_epsg = right.to_epsg()

    if left_epsg is not None and left_epsg == right_epsg:
        return True

    left_codes = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', left.to_wkt())
    right_codes = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', right.to_wkt())

    if right_epsg is not None and left_codes and left_codes[-1] == str(right_epsg):
        return True

    if left_epsg is not None and right_codes and right_codes[-1] == str(left_epsg):
        return True

    return False


def mosaic_geotiffs(
    *,
    input_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
    compression: str = "LZW",
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[postprocess] Mosaic exists, skipping: {output_path}")
        return output_path

    if not input_paths:
        raise FileNotFoundError("No GeoTIFF files provided for mosaic_geotiffs.")

    print("[postprocess] Building GeoTIFF mosaic")
    print(f"[postprocess] Input files: {len(input_paths)}")
    print(f"[postprocess] Output: {output_path}")

    srcs = []

    try:
        for path in input_paths:
            print(f"  - {path}")
            srcs.append(rasterio.open(path))

        mosaic, transform = merge(srcs)

        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            compress=compression,
            BIGTIFF="IF_SAFER",
        )

        ensure_parent(output_path)

        if output_path.exists() and overwrite:
            output_path.unlink()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
            dst.update_tags(
                postprocess="mosaic_geotiff",
                source_file_count=str(len(input_paths)),
            )

    finally:
        for src in srcs:
            src.close()

    print(f"[postprocess] Mosaic written: {output_path}")
    return output_path


def mosaic_zip_geotiffs(
    *,
    zip_paths: list[Path],
    output_path: Path,
    zip_member_pattern: str | None,
    overwrite: bool,
    compression: str = "LZW",
    allow_multiple_zip_members: bool = False,
    skip_zip_without_matching_members: bool = False,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[postprocess] Mosaic exists, skipping: {output_path}")
        return output_path

    if not zip_paths:
        raise FileNotFoundError("No ZIP files provided for mosaic_zip_geotiff.")

    raster_uris: list[str] = []

    for zip_path in zip_paths:
        members = find_tif_members_in_zip(
            zip_path=zip_path,
            zip_member_pattern=zip_member_pattern,
        )

        if len(members) == 0:
            message = (
                f"No GeoTIFF members found in {zip_path} "
                f"with zip_member_pattern={zip_member_pattern!r}"
            )

            if skip_zip_without_matching_members:
                print(f"[postprocess] {message}. Skipping ZIP.")
                continue

            raise FileNotFoundError(message)

        if len(members) > 1 and not allow_multiple_zip_members:
            listing = "\n".join(members[:30])
            raise RuntimeError(
                f"More than one GeoTIFF member found in {zip_path}.\n"
                "Please set a more restrictive zip_member_pattern, or set:\n"
                "  allow_multiple_zip_members: true\n"
                "if all matching TIFFs should be mosaicked.\n"
                f"Members:\n{listing}"
            )

        for member in members:
            raster_uris.append(zip_member_to_rasterio_uri(zip_path, member))

    if not raster_uris:
        raise FileNotFoundError(
            "No GeoTIFF members were selected for the mosaic after applying "
            f"zip_member_pattern={zip_member_pattern!r}."
        )

    print("[postprocess] Building ZIP GeoTIFF mosaic")
    print(f"[postprocess] ZIP files: {len(zip_paths)}")
    print(f"[postprocess] Raster members selected: {len(raster_uris)}")
    print(f"[postprocess] Output mosaic: {output_path}")

    srcs = []
    try:
        for uri in raster_uris:
            print(f"  - {uri}")
            srcs.append(rasterio.open(uri))

        mosaic, transform = merge(srcs)

        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            compress=compression,
            BIGTIFF="IF_SAFER",
        )

        ensure_parent(output_path)

        if output_path.exists() and overwrite:
            output_path.unlink()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
            dst.update_tags(
                postprocess="mosaic_zip_geotiff",
                source_zip_count=str(len(zip_paths)),
                selected_raster_count=str(len(raster_uris)),
            )

    finally:
        for src in srcs:
            src.close()

    print(f"[postprocess] Mosaic written: {output_path}")
    return output_path


def mosaic_mixed_geotiffs(
    *,
    input_paths: list[Path],
    output_path: Path,
    zip_member_pattern: str | None,
    overwrite: bool,
    compression: str = "LZW",
    allow_multiple_zip_members: bool = False,
    skip_zip_without_matching_members: bool = False,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[postprocess] Mosaic exists, skipping: {output_path}")
        return output_path

    raster_uris, zip_count = collect_mixed_raster_uris(
        input_paths=input_paths,
        zip_member_pattern=zip_member_pattern,
        allow_multiple_zip_members=allow_multiple_zip_members,
        skip_zip_without_matching_members=skip_zip_without_matching_members,
    )

    if not raster_uris:
        raise FileNotFoundError(
            "No GeoTIFF files or ZIP GeoTIFF members were selected for "
            "mosaic_mixed_geotiff."
        )

    print("[postprocess] Building mixed GeoTIFF mosaic")
    print(f"[postprocess] Input files: {len(input_paths)}")
    print(f"[postprocess] ZIP files inspected: {zip_count}")
    print(f"[postprocess] Raster inputs selected: {len(raster_uris)}")
    print(f"[postprocess] Output mosaic: {output_path}")

    srcs = []
    try:
        for uri in raster_uris:
            print(f"  - {uri}")
            srcs.append(rasterio.open(uri))

        mosaic, transform = merge(srcs)

        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            compress=compression,
            BIGTIFF="IF_SAFER",
        )

        ensure_parent(output_path)

        if output_path.exists() and overwrite:
            output_path.unlink()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
            dst.update_tags(
                postprocess="mosaic_mixed_geotiff",
                input_file_count=str(len(input_paths)),
                selected_raster_count=str(len(raster_uris)),
            )

    finally:
        for src in srcs:
            src.close()

    print(f"[postprocess] Mosaic written: {output_path}")
    return output_path


def mosaic_mixed_geotiffs_to_clip_aoi(
    *,
    input_paths: list[Path],
    output_path: Path,
    zip_member_pattern: str | None,
    source_cfg: dict[str, Any],
    overwrite: bool,
    compression: str = "LZW",
    allow_multiple_zip_members: bool = False,
    skip_zip_without_matching_members: bool = False,
    mem_limit_mb: int = 512,
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[postprocess] Clipped mosaic exists, skipping: {output_path}")
        return output_path

    raster_uris, zip_count = collect_mixed_raster_uris(
        input_paths=input_paths,
        zip_member_pattern=zip_member_pattern,
        allow_multiple_zip_members=allow_multiple_zip_members,
        skip_zip_without_matching_members=skip_zip_without_matching_members,
    )

    if not raster_uris:
        raise FileNotFoundError(
            "No GeoTIFF files or ZIP GeoTIFF members were selected for "
            "mosaic_mixed_geotiff_to_clip_aoi."
        )

    target_crs = _configured_source_crs(source_cfg)
    srcs = []

    try:
        if target_crs is None:
            with rasterio.open(raster_uris[0]) as first_src:
                if first_src.crs is None:
                    raise ValueError(f"Input raster has no CRS: {raster_uris[0]}")
                target_crs = first_src.crs

        clip_bounds = _clip_aoi_bounds_for_crs(
            source_cfg=source_cfg,
            target_crs=target_crs,
        )

        for uri in raster_uris:
            src = rasterio.open(uri)

            if src.crs is None:
                print(f"[postprocess] Skipping raster without CRS: {uri}")
                src.close()
                continue

            if not _crs_equivalent(src.crs, target_crs):
                print(
                    "[postprocess] Skipping raster with CRS outside target mosaic "
                    f"({src.crs} != {target_crs}): {uri}"
                )
                src.close()
                continue

            if not _bounds_intersect(*src.bounds, *clip_bounds):
                print(f"[postprocess] Skipping raster outside clip AOI: {uri}")
                src.close()
                continue

            srcs.append(src)

        if not srcs:
            raise FileNotFoundError(
                "No selected GeoTIFF intersects the clip AOI after applying CRS "
                f"and bounds filters. Clip bounds in {target_crs}: {clip_bounds}"
            )

        xres, yres = srcs[0].res
        estimated_width = int(round((clip_bounds[2] - clip_bounds[0]) / abs(xres)))
        estimated_height = int(round((clip_bounds[3] - clip_bounds[1]) / abs(yres)))

        print("[postprocess] Building clip-AOI mixed GeoTIFF mosaic")
        print(f"[postprocess] Input files: {len(input_paths)}")
        print(f"[postprocess] ZIP files inspected: {zip_count}")
        print(f"[postprocess] Raster inputs selected: {len(raster_uris)}")
        print(f"[postprocess] Raster inputs intersecting AOI: {len(srcs)}")
        print(f"[postprocess] Clip bounds ({target_crs}): {clip_bounds}")
        print(
            "[postprocess] Estimated clipped mosaic shape: "
            f"{estimated_height} x {estimated_width}"
        )
        print(f"[postprocess] Output mosaic: {output_path}")
        for src in srcs[:50]:
            print(f"  - {src.name}")

        ensure_parent(output_path)

        if output_path.exists() and overwrite:
            output_path.unlink()

        merge(
            srcs,
            bounds=clip_bounds,
            target_aligned_pixels=True,
            dst_path=output_path,
            dst_kwds={
                "driver": "GTiff",
                "compress": compression,
                "BIGTIFF": "IF_SAFER",
            },
            mem_limit=mem_limit_mb,
        )

        with rasterio.open(output_path, "r+") as dst:
            dst.update_tags(
                postprocess="mosaic_mixed_geotiff_to_clip_aoi",
                input_file_count=str(len(input_paths)),
                selected_raster_count=str(len(raster_uris)),
                intersecting_raster_count=str(len(srcs)),
                clip_bounds=",".join(str(value) for value in clip_bounds),
                clip_bounds_crs=str(target_crs),
            )

    finally:
        for src in srcs:
            src.close()

    print(f"[postprocess] Clipped mosaic written: {output_path}")
    return output_path


def run_static_postprocess(
    *,
    postprocess: str | None,
    input_paths: list[Path],
    output_path: Path,
    spec: dict[str, Any],
    source_cfg: dict[str, Any],
) -> Path:
    download_cfg = source_cfg.get("download", {}) or {}
    hda_cfg = download_cfg.get("hda", {}) or {}
    output_cfg = source_cfg.get("output", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    compression = str(
        hda_cfg.get(
            "mosaic_compression",
            output_cfg.get("compression", "LZW"),
        )
    )

    postprocess = postprocess or "copy_single"

    if postprocess in ("copy_single", "none", ""):
        return copy_single_file(
            input_paths=input_paths,
            output_path=output_path,
            overwrite=overwrite,
        )

    if postprocess == "mosaic_geotiff":
        return mosaic_geotiffs(
            input_paths=input_paths,
            output_path=output_path,
            overwrite=overwrite,
            compression=compression,
        )

    if postprocess == "mosaic_zip_geotiff":
        zip_paths = [
            path
            for path in input_paths
            if path.suffix.lower() == ".zip"
        ]

        if not zip_paths:
            raise FileNotFoundError(
                "postprocess=mosaic_zip_geotiff was requested, "
                "but no ZIP files were provided."
            )

        return mosaic_zip_geotiffs(
            zip_paths=zip_paths,
            output_path=output_path,
            zip_member_pattern=spec.get("zip_member_pattern"),
            overwrite=overwrite,
            compression=compression,
            allow_multiple_zip_members=bool(
                spec.get("allow_multiple_zip_members", False)
            ),
            skip_zip_without_matching_members=bool(
                spec.get("skip_zip_without_matching_members", False)
            ),
        )

    if postprocess == "mosaic_mixed_geotiff":
        return mosaic_mixed_geotiffs(
            input_paths=input_paths,
            output_path=output_path,
            zip_member_pattern=spec.get("zip_member_pattern"),
            overwrite=overwrite,
            compression=compression,
            allow_multiple_zip_members=bool(
                spec.get("allow_multiple_zip_members", False)
            ),
            skip_zip_without_matching_members=bool(
                spec.get("skip_zip_without_matching_members", False)
            ),
        )

    if postprocess == "mosaic_mixed_geotiff_to_clip_aoi":
        return mosaic_mixed_geotiffs_to_clip_aoi(
            input_paths=input_paths,
            output_path=output_path,
            zip_member_pattern=spec.get("zip_member_pattern"),
            source_cfg=source_cfg,
            overwrite=overwrite,
            compression=compression,
            allow_multiple_zip_members=bool(
                spec.get("allow_multiple_zip_members", False)
            ),
            skip_zip_without_matching_members=bool(
                spec.get("skip_zip_without_matching_members", False)
            ),
            mem_limit_mb=int(hda_cfg.get("mosaic_mem_limit_mb", 512)),
        )

    raise NotImplementedError(
        f"Unsupported static postprocess={postprocess!r}."
    )
