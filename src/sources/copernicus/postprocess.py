from __future__ import annotations

from pathlib import Path
import re
import shutil
import zipfile
from typing import Any

import rasterio
from rasterio.merge import merge


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

    raise NotImplementedError(
        f"Unsupported static postprocess={postprocess!r}."
    )