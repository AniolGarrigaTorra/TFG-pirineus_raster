from __future__ import annotations

from pathlib import Path

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_clipped_dir
from src.pipeline.feature_writer import write_feature_raster
from src.pipeline.grid_context import load_grid_context, print_grid_context
from src.pipeline.metadata import build_static_feature_metadata
from src.pipeline.raster_reading import read_raster_to_grid
from src.pipeline.resampling import (
    get_variable_resampling_method,
    get_variable_resampling_method_name,
)
from src.pipeline.source_config import get_static_layer_items
from src.sources.worldclim.naming import (
    build_worldclim_clipped_name,
    build_worldclim_feature_name,
)


def _get_static_layer_structure(source_cfg: dict) -> str:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure not in {"static_single", "static_index_set"}:
        raise ValueError(
            "WorldClim static builder only supports "
            f"static_single or static_index_set, got: {layer_structure}"
        )

    return layer_structure


def _get_static_clipped_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    layer_name: str,
) -> Path:
    """
    Build the expected clipped raster path for one static WorldClim layer.

    Supports:
      - static_single: elev
      - static_index_set: bio1...bio19
    """
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]

    clipped_dir = get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=clip_aoi_name,
        source_resolution=source_resolution,
        variable=layer_name,
    )

    clipped_name = build_worldclim_clipped_name(
        source_cfg=source_cfg,
        layer_name=layer_name,
        domain_name=clip_aoi_name,
    )

    return clipped_dir / clipped_name


def _get_static_output_path(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
    layer_name: str,
) -> Path:
    source = source_cfg["source"]

    provider = source["provider"]
    product = source["product"]

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

    output_name = build_worldclim_feature_name(
        provider=provider,
        product=product,
        variable=layer_name,
        metric=None,
        start_month=None,
        end_month=None,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return output_dir / output_name


def build_worldclim_static_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned static WorldClim features.

    Supports:
      - static_index_set: bioclimatic indices bio1...bio19
      - static_single: elevation

    This implementation uses the generic pipeline helpers:
      - GridContext
      - read_raster_to_grid
      - write_feature_raster

    Provider-specific logic is restricted to:
      - WorldClim clipped filename conventions
      - WorldClim output filename conventions
      - WorldClim static metadata
    """
    _get_static_layer_structure(source_cfg)

    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {})

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]

    target_resolution_m = int(processing["target_resolution_m"])

    nodata = float(project_cfg.get("nodata", -9999.0))
    output_dtype = output_cfg.get("dtype", "float32")
    compression = output_cfg.get("compression", "LZW")
    write_sidecar = bool(output_cfg.get("write_sidecar_json", True))

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    print("[build-static] Output AOI:", output_aoi_name)
    print("[build-static] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-static]")
    print(
        "[build-static] Layer structure:",
        source_cfg.get("dataset", {}).get("layer_structure"),
    )

    written_paths: list[Path] = []

    for layer_name, layer_cfg in get_static_layer_items(source_cfg):
        scale_factor = float(layer_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(
            source_cfg=source_cfg,
            variable=layer_name,
        )
        resampling_name = get_variable_resampling_method_name(
            source_cfg=source_cfg,
            variable=layer_name,
        )

        clipped_path = _get_static_clipped_path(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            layer_name=layer_name,
        )

        if not clipped_path.exists():
            raise FileNotFoundError(
                f"Missing clipped static raster: {clipped_path}"
            )

        output_path = _get_static_output_path(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            layer_name=layer_name,
        )

        print("==============================")
        print(f"[build-static] Layer: {layer_name}")
        print(f"[build-static] Description: {layer_cfg.get('description', '')}")
        print(f"[build-static] Scale factor: {scale_factor}")
        print(f"[build-static] Resampling: {resampling_name}")
        print(f"[build-static] Clipped path: {clipped_path}")
        print(f"[build-static] Output path: {output_path}")

        grid_array = read_raster_to_grid(
            raster_path=clipped_path,
            grid=grid,
            resampling=resampling,
            band=1,
            scale_factor=scale_factor,
        )

        metadata = build_static_feature_metadata(
            source_cfg=source_cfg,
            layer_name=layer_name,
            layer_cfg=layer_cfg,
            clip_aoi_name=clip_aoi_name,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            resampling_method_name=resampling_name,
        )

        written_path = write_feature_raster(
            output_path=output_path,
            array=grid_array,
            grid=grid,
            metadata=metadata,
            output_dtype=output_dtype,
            nodata=nodata,
            compression=compression,
            write_sidecar=write_sidecar,
            validate=True,
        )

        print(f"[build-static] Written: {written_path}")
        written_paths.append(written_path)

    return written_paths