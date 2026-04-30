from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_clipped_dir
from src.pipeline.aggregation import aggregate_stack, months_from_range
from src.pipeline.feature_writer import write_feature_raster
from src.pipeline.grid_context import load_grid_context, print_grid_context
from src.pipeline.metadata import build_feature_metadata
from src.pipeline.raster_reading import read_raster_to_grid
from src.pipeline.resampling import (
    get_variable_resampling_method,
    get_variable_resampling_method_name,
)
from src.pipeline.source_config import (
    aggregation_applies_to_variable,
    get_enabled_variable_items,
    get_temporal_aggregations,
)
from src.sources.worldclim.naming import (
    build_worldclim_clipped_name,
    build_worldclim_feature_name,
)


def _ensure_monthly_climatology(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure != "monthly_climatology":
        raise ValueError(
            "WorldClim monthly climatology builder only supports "
            f"layer_structure='monthly_climatology', got: {layer_structure}"
        )


def _get_monthly_clipped_dir(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
) -> Path:
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]

    return get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=clip_aoi_name,
        source_resolution=source_resolution,
        variable=variable,
    )


def _get_monthly_clipped_path(
    clipped_dir: Path,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
    month: int,
) -> Path:
    clipped_name = build_worldclim_clipped_name(
        source_cfg=source_cfg,
        layer_name=variable,
        domain_name=clip_aoi_name,
        month=month,
    )

    return clipped_dir / clipped_name


def _get_monthly_output_path(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
    variable: str,
    metric: str,
    months: list[int],
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
        variable=variable,
        metric=metric,
        start_month=min(months),
        end_month=max(months),
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return output_dir / output_name


def _read_monthly_stack_to_grid(
    clipped_dir: Path,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
    months: list[int],
    grid,
    resampling,
    scale_factor: float,
) -> np.ndarray:
    monthly_arrays: list[np.ndarray] = []

    for month in months:
        clipped_path = _get_monthly_clipped_path(
            clipped_dir=clipped_dir,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            variable=variable,
            month=month,
        )

        if not clipped_path.exists():
            raise FileNotFoundError(
                f"Missing clipped monthly raster for variable={variable}, "
                f"month={month}: {clipped_path}"
            )

        monthly_grid = read_raster_to_grid(
            raster_path=clipped_path,
            grid=grid,
            resampling=resampling,
            band=1,
            scale_factor=scale_factor,
        )

        monthly_arrays.append(monthly_grid)

    if not monthly_arrays:
        raise ValueError(
            f"No monthly arrays collected for variable={variable}, months={months}"
        )

    return np.stack(monthly_arrays, axis=0)


def build_worldclim_monthly_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned features from WorldClim monthly climatologies.

    This supports products with:
      dataset.layer_structure: monthly_climatology

    Typical input:
      wc2.1_10m_tmin_01_pyrenees_full.tif
      wc2.1_10m_tmin_02_pyrenees_full.tif
      ...

    Typical output:
      worldclim_v2_1_climate_normals_tmin_annual_mean_experimental_pallars_sobira_100m.tif
      worldclim_v2_1_climate_normals_prec_may_sep_sum_experimental_pallars_sobira_100m.tif
    """
    _ensure_monthly_climatology(source_cfg)

    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {})

    provider = source["provider"]
    product = source["product"]

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

    print("[build-monthly] Output AOI:", output_aoi_name)
    print("[build-monthly] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-monthly]")
    print("[build-monthly] Layer structure:", source_cfg["dataset"]["layer_structure"])

    aggregations = get_temporal_aggregations(source_cfg)
    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(
            source_cfg=source_cfg,
            variable=variable,
        )
        resampling_name = get_variable_resampling_method_name(
            source_cfg=source_cfg,
            variable=variable,
        )

        clipped_dir = _get_monthly_clipped_dir(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            variable=variable,
        )

        if not clipped_dir.exists():
            raise FileNotFoundError(
                f"Clipped directory not found for variable '{variable}': {clipped_dir}"
            )

        print("==============================")
        print(f"[build-monthly] Variable: {variable}")
        print(f"[build-monthly] Product: {provider}/{product}")
        print(f"[build-monthly] Scale factor: {scale_factor}")
        print(f"[build-monthly] Resampling: {resampling_name}")
        print(f"[build-monthly] Clipped dir: {clipped_dir}")

        for aggregation_cfg in aggregations:
            if not aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            metric = aggregation_cfg["metric"]
            months = months_from_range(aggregation_cfg["months"])

            print(
                f"[build-monthly] Aggregation: "
                f"{aggregation_cfg.get('name', metric)} "
                f"metric={metric}, months={months}"
            )

            stack = _read_monthly_stack_to_grid(
                clipped_dir=clipped_dir,
                source_cfg=source_cfg,
                clip_aoi_name=clip_aoi_name,
                variable=variable,
                months=months,
                grid=grid,
                resampling=resampling,
                scale_factor=scale_factor,
            )

            aggregated = aggregate_stack(
                stack=stack,
                metric=metric,
            ).astype(np.float32)

            output_path = _get_monthly_output_path(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                variable=variable,
                metric=metric,
                months=months,
            )

            metadata = build_feature_metadata(
                source_cfg=source_cfg,
                variable=variable,
                variable_cfg=variable_cfg,
                aggregation_cfg=aggregation_cfg,
                months=months,
                clip_aoi_name=clip_aoi_name,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                resampling_method_name=resampling_name,
            )

            written_path = write_feature_raster(
                output_path=output_path,
                array=aggregated,
                grid=grid,
                metadata=metadata,
                output_dtype=output_dtype,
                nodata=nodata,
                compression=compression,
                write_sidecar=write_sidecar,
                validate=True,
            )

            print(f"[build-monthly] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths