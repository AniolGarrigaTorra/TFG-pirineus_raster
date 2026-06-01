from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize

gpd.options.io_engine = "fiona"

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_interim_dir
from src.pipeline.progress import progress_log
from src.pipeline.raster_ops import (
    load_grid_context,
    print_grid_context,
    write_feature_raster,
)
from src.sources.igme_brgm.naming import (
    build_igme_brgm_feature_name,
    build_igme_brgm_legend_name,
    safe_name,
)


def _get_target_resolution_m(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


def _get_source_resolution(source_cfg: dict) -> str:
    return str(source_cfg["processing"]["source_resolution"])


def _load_target_grid(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_cfg: dict,
):
    return load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=_get_target_resolution_m(source_cfg),
    )


def _get_output_options(
    project_cfg: dict,
    source_cfg: dict,
) -> dict:
    output_cfg = source_cfg.get("output", {})

    return {
        "nodata": int(project_cfg.get("nodata", -9999)),
        "output_dtype": output_cfg.get("dtype", "int32"),
        "compression": output_cfg.get("compression", "LZW"),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _get_feature_output_dir(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
) -> Path:
    source = source_cfg["source"]

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    ensure_dir(output_dir)

    return output_dir


def _get_clipped_vector_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    dataset_name: str,
    layer_name: str,
) -> Path:
    source = source_cfg["source"]

    return (
        get_source_interim_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
        )
        / "clipped"
        / clip_aoi_name
        / _get_source_resolution(source_cfg)
        / safe_name(dataset_name)
        / f"igme_brgm_{safe_name(dataset_name)}_{safe_name(layer_name)}_{safe_name(clip_aoi_name)}.gpkg"
    )


def _read_clipped_layer(path: Path, layer_name: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clipped vector layer not found: {path}\n"
            "Run the clip stage first."
        )

    return gpd.read_file(
        path,
        layer=safe_name(layer_name),
        engine="fiona",
    )


def _build_source_metadata(
    source_cfg: dict,
    feature_cfg: dict,
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
) -> dict[str, Any]:
    source = source_cfg["source"]

    metadata = {
        "source_id": source.get("id"),
        "provider": source.get("provider"),
        "product": source.get("product"),
        "source_config_path": source_cfg.get("_config_path"),
        "product_group": source.get("product_group"),
        "version": source.get("version"),
        "description": source.get("description"),
        "page_url": source.get("page_url"),
        "citation": source.get("citation"),
        "source_scale": source.get("source_scale"),
        "source_crs": source.get("source_crs"),
        "variable": feature_cfg["name"],
        "variable_description": feature_cfg.get("description"),
        "unit": feature_cfg.get("unit", "category_code"),
        "dataset": feature_cfg.get("dataset"),
        "source_layer": feature_cfg.get("layer"),
        "value_field": feature_cfg.get("value_field"),
        "encode_categories": bool(feature_cfg.get("encode_categories", False)),
        "clip_aoi_name": clip_aoi_name,
        "output_aoi_name": output_aoi_name,
        "target_resolution_m": target_resolution_m,
        "resampling": "none_vector_rasterization",
        "rasterization_method": "rasterio.features.rasterize",
        "all_touched": bool(feature_cfg.get("all_touched", False)),
    }

    return {k: v for k, v in metadata.items() if v is not None}


def _prepare_numeric_values(
    gdf: gpd.GeoDataFrame,
    value_field: str,
) -> tuple[gpd.GeoDataFrame, str, pd.DataFrame]:
    gdf = gdf.copy()

    values = pd.to_numeric(gdf[value_field], errors="coerce")
    gdf["_raster_value"] = values

    valid = gdf["_raster_value"].notna()
    gdf = gdf[valid].copy()

    gdf["_raster_value"] = gdf["_raster_value"].astype("int32")

    legend = (
        gdf.drop(columns="geometry")
        .drop_duplicates(subset=["_raster_value"])
        .sort_values("_raster_value")
        .copy()
    )

    return gdf, "_raster_value", legend


def _prepare_encoded_values(
    gdf: gpd.GeoDataFrame,
    value_field: str,
) -> tuple[gpd.GeoDataFrame, str, pd.DataFrame]:
    gdf = gdf.copy()

    original_values = (
        gdf[value_field]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    mapping = {
        original_value: idx
        for idx, original_value in enumerate(original_values, start=1)
    }

    gdf["_original_category"] = gdf[value_field].astype(str)
    gdf["_raster_value"] = gdf["_original_category"].map(mapping)

    valid = gdf["_raster_value"].notna()
    gdf = gdf[valid].copy()
    gdf["_raster_value"] = gdf["_raster_value"].astype("int32")

    legend = pd.DataFrame(
        {
            "_raster_value": list(mapping.values()),
            value_field: list(mapping.keys()),
        }
    ).sort_values("_raster_value")

    return gdf, "_raster_value", legend


def _select_legend_columns(
    legend: pd.DataFrame,
    feature_cfg: dict,
    value_field: str,
) -> pd.DataFrame:
    legend_fields = feature_cfg.get("legend_fields", [])

    desired = ["_raster_value", value_field] + list(legend_fields)
    existing = [col for col in desired if col in legend.columns]

    if not existing:
        return legend

    legend = legend[existing].copy()
    legend = legend.drop_duplicates()

    return legend


def _write_legend_csv(
    legend: pd.DataFrame,
    output_dir: Path,
    source_cfg: dict,
    feature_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
) -> Path:
    source = source_cfg["source"]

    legend_path = output_dir / build_igme_brgm_legend_name(
        provider=source["provider"],
        product=source["product"],
        feature_name=feature_cfg["name"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    legend.to_csv(
        legend_path,
        index=False,
        encoding="utf-8",
    )

    progress_log(f"[build] Legend written: {legend_path}")

    return legend_path


def _rasterize_geodataframe(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    grid,
    nodata: int,
    all_touched: bool,
) -> np.ndarray:
    if gdf.empty:
        return np.full(grid.shape, nodata, dtype=np.int32)

    shapes = (
        (geom, int(value))
        for geom, value in zip(gdf.geometry, gdf[value_column])
        if geom is not None and not geom.is_empty and pd.notna(value)
    )

    array = rasterize(
        shapes=shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=nodata,
        dtype="int32",
        all_touched=all_touched,
    )

    return array.astype(np.int32)


def _build_one_feature(
    project_cfg: dict,
    source_cfg: dict,
    feature_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
    grid,
    output_dir: Path,
    output_options: dict,
) -> list[Path]:
    source = source_cfg["source"]

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)

    dataset_name = feature_cfg["dataset"]
    layer_name = feature_cfg["layer"]
    value_field = feature_cfg["value_field"]

    clipped_path = _get_clipped_vector_path(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_name=clip_aoi_name,
        dataset_name=dataset_name,
        layer_name=layer_name,
    )

    gdf = _read_clipped_layer(
        path=clipped_path,
        layer_name=layer_name,
    )

    if gdf.crs is None:
        raise ValueError(f"Clipped layer has no CRS: {clipped_path}")

    if gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)

    if value_field not in gdf.columns:
        raise ValueError(
            f"Field '{value_field}' not found in clipped layer {clipped_path}.\n"
            f"Available columns: {list(gdf.columns)}"
        )

    encode_categories = bool(feature_cfg.get("encode_categories", False))

    if encode_categories:
        gdf, value_column, legend = _prepare_encoded_values(
            gdf=gdf,
            value_field=value_field,
        )
    else:
        gdf, value_column, legend = _prepare_numeric_values(
            gdf=gdf,
            value_field=value_field,
        )

    legend = _select_legend_columns(
        legend=legend,
        feature_cfg=feature_cfg,
        value_field=value_field,
    )

    raster = _rasterize_geodataframe(
        gdf=gdf,
        value_column=value_column,
        grid=grid,
        nodata=int(output_options["nodata"]),
        all_touched=bool(feature_cfg.get("all_touched", False)),
    )

    output_name = build_igme_brgm_feature_name(
        provider=source["provider"],
        product=source["product"],
        feature_name=feature_cfg["name"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_path = output_dir / output_name

    metadata = _build_source_metadata(
        source_cfg=source_cfg,
        feature_cfg=feature_cfg,
        clip_aoi_name=clip_aoi_name,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    progress_log(f"[build] Rasterizing feature: {feature_cfg['name']}")
    progress_log(f"  Dataset: {dataset_name}")
    progress_log(f"  Layer:   {layer_name}")
    progress_log(f"  Field:   {value_field}")
    progress_log(f"  Out:     {output_path}")

    write_feature_raster(
        output_path=output_path,
        array=raster,
        grid=grid,
        metadata=metadata,
        output_dtype=output_options["output_dtype"],
        nodata=output_options["nodata"],
        compression=output_options["compression"],
        write_sidecar=output_options["write_sidecar"],
        validate=True,
    )

    legend_path = _write_legend_csv(
        legend=legend,
        output_dir=output_dir,
        source_cfg=source_cfg,
        feature_cfg=feature_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return [output_path, legend_path]


def build_igme_brgm_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned geological categorical rasters.
    """
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_aoi_name = output_aoi_cfg["name"]

    grid = _load_target_grid(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )

    print_grid_context(grid)

    output_dir = _get_feature_output_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_options = _get_output_options(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
    )

    features = source_cfg.get("features", [])

    if not features:
        raise ValueError("No features configured under source_cfg['features'].")

    output_paths: list[Path] = []

    for feature_cfg in features:
        if not feature_cfg.get("enabled", True):
            continue

        paths = _build_one_feature(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            feature_cfg=feature_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
            grid=grid,
            output_dir=output_dir,
            output_options=output_options,
        )

        output_paths.extend(paths)

    return output_paths
