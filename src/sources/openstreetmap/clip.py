from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import fiona
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

from src.io.paths import ensure_dir, get_source_clipped_dir, get_source_raw_dir
from src.pipeline.progress import progress_log
from src.sources.openstreetmap.naming import (
    build_osm_clipped_name,
    build_osm_raw_path,
    get_enabled_layer_items,
    get_source_resolution,
    validate_osm_source_config,
)


OTHER_TAG_RE = re.compile(r'"([^"]+)"=>"([^"]*)"')


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


def _parse_other_tags(value: Any) -> dict[str, str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    return dict(OTHER_TAG_RE.findall(str(value)))


def _tag_value(row: pd.Series, key: str) -> str | None:
    if key in row.index:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value) != "":
            return str(value)
    other_tags = _parse_other_tags(row.get("other_tags"))
    return other_tags.get(key)


def _matches_tags(row: pd.Series, tags: dict[str, Any]) -> bool:
    for key, expected in tags.items():
        value = _tag_value(row, key)
        if value is None:
            return False
        if isinstance(expected, list):
            if value not in [str(item) for item in expected]:
                return False
        elif expected is True:
            if value in {"no", "false", "0"}:
                return False
        elif str(value) != str(expected):
            return False
    return True


def _read_osm_layer(
    pbf_path: Path,
    osm_layer: str,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(
            pbf_path,
            layer=osm_layer,
            bbox=bbox_wgs84,
            engine="pyogrio",
        )
    except Exception as exc:
        message = str(exc)
        if "Null layer" in message or "not recognized as being in a supported" in message:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        raise RuntimeError(
            f"Could not read OSM PBF layer '{osm_layer}' from {pbf_path}. "
            "OpenStreetMap PBF clipping requires a working pyogrio/GDAL/PROJ "
            "stack. Rebuild the environment from environment.yml or install "
            "pyogrio from conda-forge so it matches the GDAL/PROJ libraries."
        ) from exc


def _clip_layer_from_regions(
    *,
    source_cfg: dict,
    layer_key: str,
    layer_cfg: dict,
    raw_dir: Path,
    clip_geom: gpd.GeoDataFrame,
    bbox_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    regions = source_cfg.get("download", {}).get("regions", []) or []
    osm_layers = layer_cfg.get("osm_layers", ["lines"])
    tags = layer_cfg.get("tags", {}) or {}

    for region_cfg in regions:
        pbf_path = build_osm_raw_path(raw_dir, region_cfg)
        if not pbf_path.exists():
            raise FileNotFoundError(f"Missing OSM PBF: {pbf_path}")

        for osm_layer in osm_layers:
            progress_log(f"[clip:osm] Reading {pbf_path.name} layer={osm_layer} for {layer_key}")
            gdf = _read_osm_layer(pbf_path, str(osm_layer), bbox_wgs84)
            if gdf.empty:
                continue

            if tags:
                mask = gdf.apply(lambda row: _matches_tags(row, tags), axis=1)
                gdf = gdf[mask]
            if gdf.empty:
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            frames.append(gdf.to_crs(clip_geom.crs))

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=clip_geom.crs)

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=clip_geom.crs)
    combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty]
    if combined.empty:
        return combined

    invalid_mask = ~combined.geometry.is_valid
    if invalid_mask.any():
        progress_log(
            f"[clip:osm] Repairing invalid geometries: {int(invalid_mask.sum())}"
        )
        combined = combined.copy()
        combined.loc[invalid_mask, "geometry"] = combined.loc[
            invalid_mask,
            "geometry",
        ].make_valid()
        combined = combined.explode(ignore_index=True)
        combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty]
        if combined.empty:
            return combined

    clipped = gpd.clip(combined, clip_geom)
    return clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]


def _write_gpkg(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    if not gdf.empty:
        gdf.to_file(output_path, layer="features", driver="GPKG")
        return

    schema = {
        "geometry": "Unknown",
        "properties": {"empty": "int"},
    }
    with fiona.open(
        output_path,
        mode="w",
        driver="GPKG",
        layer="features",
        crs=gdf.crs,
        schema=schema,
    ):
        pass


def _allow_empty_layer(layer_cfg: dict[str, Any]) -> bool:
    return bool(layer_cfg.get("allow_empty", False))


def clip_osm_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    validate_osm_source_config(source_cfg)

    source = source_cfg["source"]
    processing = source_cfg.get("processing", {}) or {}
    source_resolution = get_source_resolution(source_cfg)
    clip_aoi_name = clip_aoi_cfg["name"]
    clip_bounds = _get_aoi_bounds(clip_aoi_cfg)
    clip_crs = str(clip_aoi_cfg["crs"])
    bbox_wgs84 = _transform_bounds(clip_bounds, src_crs=clip_crs, dst_crs="EPSG:4326")

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=source_resolution,
    )

    clip_geom = gpd.GeoDataFrame(
        geometry=[box(*clip_bounds)],
        crs=clip_crs,
    )

    overwrite = bool(source_cfg.get("download", {}).get("overwrite_existing", False))
    written_paths: list[Path] = []

    progress_log(f"[clip:osm] Provider: {source['provider']}")
    progress_log(f"[clip:osm] Product: {source['product']}")
    progress_log(f"[clip:osm] AOI: {clip_aoi_name}")
    progress_log(f"[clip:osm] Raw dir: {raw_dir}")

    for layer_key, layer_cfg in get_enabled_layer_items(source_cfg):
        output_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=layer_key.replace(".", "_"),
        )
        output_path = output_dir / build_osm_clipped_name(layer_key, clip_aoi_name)

        if output_path.exists() and not overwrite:
            progress_log(f"[clip:osm] Exists, skipping: {output_path}")
            written_paths.append(output_path)
            continue

        ensure_dir(output_path.parent)
        clipped = _clip_layer_from_regions(
            source_cfg=source_cfg,
            layer_key=layer_key,
            layer_cfg=layer_cfg,
            raw_dir=raw_dir,
            clip_geom=clip_geom,
            bbox_wgs84=bbox_wgs84,
        )

        progress_log(f"[clip:osm] Layer: {layer_key}")
        progress_log(f"[clip:osm] Features: {len(clipped)}")
        progress_log(f"[clip:osm] Output: {output_path}")

        if clipped.empty and not _allow_empty_layer(layer_cfg):
            raise ValueError(
                f"OSM layer '{layer_key}' clipped to zero features. "
                "Refusing to cache an empty enabled layer; check OSM reader, "
                "tags, AOI bounds, or set allow_empty=true if this is expected."
            )

        _write_gpkg(clipped, output_path)
        written_paths.append(output_path)

    return written_paths
