from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.merge import merge


def mosaic_geotiffs(
    *,
    input_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
    compression: str = "LZW",
) -> Path:
    """
    Build a single GeoTIFF mosaic from multiple GeoTIFF inputs.

    This is useful for tiled products such as Copernicus DEM GLO-30 COGs.
    """
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

        output_path.parent.mkdir(parents=True, exist_ok=True)

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