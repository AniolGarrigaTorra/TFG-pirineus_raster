from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage

from src.io.paths import get_feature_output_dir, get_source_clipped_dir
from src.pipeline.progress import progress_log
from src.pipeline.raster_ops import (
    build_static_feature_metadata,
    feature_raster_is_ready,
    load_grid_context,
    print_grid_context,
    write_feature_raster,
)
from src.sources.openstreetmap.naming import (
    build_osm_clipped_name,
    build_osm_feature_name,
    get_enabled_layer_items,
    get_source_resolution,
    validate_osm_source_config,
)


def _target_resolution_m(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


def _output_options(project_cfg: dict, source_cfg: dict) -> dict[str, Any]:
    output_cfg = source_cfg.get("output", {}) or {}
    return {
        "output_dtype": str(output_cfg.get("dtype", "float32")),
        "nodata": float(project_cfg.get("nodata", -9999.0)),
        "compression": str(output_cfg.get("compression", "LZW")),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _read_clipped_features(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing clipped OSM layer: {path}")
    gdf = gpd.read_file(path, layer="features")
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)
    if gdf.crs is None:
        raise ValueError(f"Clipped OSM layer has no CRS: {path}")
    return gdf.to_crs(target_crs)


def _rasterize_presence(gdf: gpd.GeoDataFrame, grid) -> np.ndarray:
    if gdf.empty:
        return np.zeros(grid.shape, dtype=np.float32)

    shapes = [
        (geom, 1.0)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        return np.zeros(grid.shape, dtype=np.float32)

    return rasterize(
        shapes=shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )


def _distance_from_presence(presence: np.ndarray, pixel_size_m: int) -> np.ndarray:
    target = np.isfinite(presence) & (presence > 0)
    if not target.any():
        return np.full(presence.shape, np.nan, dtype=np.float32)
    return (ndimage.distance_transform_edt(~target) * float(pixel_size_m)).astype(np.float32)


def _build_layer_array(gdf: gpd.GeoDataFrame, layer_cfg: dict, grid) -> tuple[np.ndarray, str]:
    output_mode = str(layer_cfg.get("output", "presence"))
    presence = _rasterize_presence(gdf, grid)

    if output_mode == "presence":
        return presence, "presence"

    if output_mode == "distance":
        return _distance_from_presence(presence, grid.resolution_m), "distance"

    if output_mode == "log10_distance":
        distance = _distance_from_presence(presence, grid.resolution_m)
        return np.log10(distance + 1.0).astype(np.float32), "log10_distance"

    raise NotImplementedError(
        f"Unsupported OSM layer output={output_mode!r}. "
        "Supported outputs: presence, distance, log10_distance"
    )


def build_osm_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    validate_osm_source_config(source_cfg)

    source = source_cfg["source"]
    source_resolution = get_source_resolution(source_cfg)
    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _target_resolution_m(source_cfg)
    output_options = _output_options(project_cfg, source_cfg)

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    progress_log(f"[build:osm] Provider: {source['provider']}")
    progress_log(f"[build:osm] Product: {source['product']}")
    progress_log(f"[build:osm] Output AOI: {output_aoi_name}")
    progress_log(f"[build:osm] Clip AOI: {clip_aoi_name}")
    print_grid_context(grid, prefix="[build:osm]")

    written_paths: list[Path] = []

    for layer_key, layer_cfg in get_enabled_layer_items(source_cfg):
        variable = layer_key.replace(".", "_")
        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=variable,
        )
        clipped_path = clipped_dir / build_osm_clipped_name(layer_key, clip_aoi_name)

        output_dir = get_feature_output_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
            domain_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
        )
        output_path = output_dir / build_osm_feature_name(
            layer_key=layer_key,
            domain_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
        )

        progress_log(f"[build:osm] Layer: {layer_key}")
        progress_log(f"[build:osm] Output mode: {layer_cfg.get('output', 'presence')}")
        progress_log(f"[build:osm] Clipped path: {clipped_path}")
        progress_log(f"[build:osm] Output path: {output_path}")

        if feature_raster_is_ready(
            output_path,
            grid,
            require_sidecar=output_options["write_sidecar"],
        ):
            progress_log(f"[build:osm] Cache hit: {layer_key} -> {output_path}")
            written_paths.append(output_path)
            continue

        gdf = _read_clipped_features(clipped_path, target_crs=str(grid.crs))
        array, output_mode = _build_layer_array(gdf, layer_cfg, grid)

        variable_cfg = {
            "description": layer_cfg.get("description"),
            "unit": layer_cfg.get("unit") or ("m" if "distance" in output_mode else "binary"),
            "valid_range": layer_cfg.get("valid_range"),
            "data_type": layer_cfg.get("data_type") or "continuous",
            "value_semantics": layer_cfg.get("value_semantics")
            or ("intensive" if "distance" in output_mode else "categorical"),
            "scale_factor": 1.0,
            "native_resolution_m": None,
        }

        metadata = build_static_feature_metadata(
            source_cfg=source_cfg,
            layer_name=variable,
            layer_cfg=variable_cfg,
            clip_aoi_name=clip_aoi_name,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            resampling_method_name="rasterize",
        )
        metadata.update(
            {
                "osm_layer_key": layer_key,
                "osm_source_layers": layer_cfg.get("osm_layers"),
                "osm_tags": layer_cfg.get("tags"),
                "osm_output_mode": output_mode,
                "feature_count": int(len(gdf)),
            }
        )

        written_paths.append(
            write_feature_raster(
                output_path=output_path,
                array=array,
                grid=grid,
                metadata=metadata,
                **output_options,
                validate=True,
            )
        )

    return written_paths
