from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_clipped_dir
from src.pipeline.config import (
    aggregation_applies_to_variable,
    get_enabled_variable_items,
    get_static_layer_items,
    get_temporal_aggregations,
    years_from_range,
)
from src.pipeline.raster_ops import (
    build_feature_metadata,
    build_static_feature_metadata,
    get_variable_resampling_method,
    get_variable_resampling_method_name,
    load_grid_context,
    print_grid_context,
    read_raster_to_grid,
    write_feature_raster,
)
from src.pipeline.temporal import (
    TemporalRasterSpec,
    aggregate_stack,
    aggregate_temporal_specs,
    aggregate_time_series,
    months_from_range,
)
from src.sources.worldclim.naming import (
    build_worldclim_clipped_name,
    build_worldclim_cmip6_clipped_month_name,
    build_worldclim_cmip6_feature_name,
    build_worldclim_feature_name,
    get_file_specs,
)


# =============================================================================
# Shared WorldClim build helpers
# =============================================================================


def _get_output_options(
    project_cfg: dict,
    source_cfg: dict,
) -> dict:
    output_cfg = source_cfg.get("output", {})

    return {
        "nodata": float(project_cfg.get("nodata", -9999.0)),
        "output_dtype": output_cfg.get("dtype", "float32"),
        "compression": output_cfg.get("compression", "LZW"),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _get_target_resolution_m(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


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


def _get_worldclim_feature_output_dir(
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


def _get_worldclim_clipped_base_dir(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
) -> Path:
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    return get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=clip_aoi_name,
        source_resolution=processing["source_resolution"],
        variable=variable,
    )


def _get_variable_processing(
    source_cfg: dict,
    variable: str,
    variable_cfg: dict,
) -> tuple[float, object, str]:
    scale_factor = float(variable_cfg.get("scale_factor", 1.0))

    resampling = get_variable_resampling_method(
        source_cfg=source_cfg,
        variable=variable,
    )
    resampling_name = get_variable_resampling_method_name(
        source_cfg=source_cfg,
        variable=variable,
    )

    return scale_factor, resampling, resampling_name


# =============================================================================
# Static products: elevation and bioclim
# =============================================================================


def _ensure_static_product(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure not in {"static_single", "static_index_set"}:
        raise ValueError(
            "WorldClim static builder only supports static_single or "
            f"static_index_set, got: {layer_structure}"
        )


def _get_static_clipped_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    layer_name: str,
) -> Path:
    clipped_dir = _get_worldclim_clipped_base_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_name=clip_aoi_name,
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

    output_dir = _get_worldclim_feature_output_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_name = build_worldclim_feature_name(
        provider=source["provider"],
        product=source["product"],
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
      - static_single: elevation
      - static_index_set: bioclimatic indices
    """
    _ensure_static_product(source_cfg)

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = _load_target_grid(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_cfg=output_aoi_cfg,
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
        scale_factor, resampling, resampling_name = _get_variable_processing(
            source_cfg=source_cfg,
            variable=layer_name,
            variable_cfg=layer_cfg,
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
            **output_options,
            validate=True,
        )

        print(f"[build-static] Written: {written_path}")
        written_paths.append(written_path)

    return written_paths


# =============================================================================
# Monthly climatology products
# =============================================================================


def _ensure_monthly_climatology(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure != "monthly_climatology":
        raise ValueError(
            "WorldClim monthly climatology builder only supports "
            f"layer_structure='monthly_climatology', got: {layer_structure}"
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

    output_dir = _get_worldclim_feature_output_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_name = build_worldclim_feature_name(
        provider=source["provider"],
        product=source["product"],
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
    """
    _ensure_monthly_climatology(source_cfg)

    source = source_cfg["source"]

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = _load_target_grid(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )

    print("[build-monthly] Output AOI:", output_aoi_name)
    print("[build-monthly] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-monthly]")
    print("[build-monthly] Layer structure:", source_cfg["dataset"]["layer_structure"])

    aggregations = get_temporal_aggregations(source_cfg)
    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        scale_factor, resampling, resampling_name = _get_variable_processing(
            source_cfg=source_cfg,
            variable=variable,
            variable_cfg=variable_cfg,
        )

        clipped_dir = _get_worldclim_clipped_base_dir(
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
        print(f"[build-monthly] Product: {source['provider']}/{source['product']}")
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
                **output_options,
                validate=True,
            )

            print(f"[build-monthly] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths


# =============================================================================
# Monthly time series products: CRU-TS
# =============================================================================


def _ensure_monthly_time_series(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure != "monthly_time_series":
        raise ValueError(
            "WorldClim monthly time series builder only supports "
            f"layer_structure='monthly_time_series', got: {layer_structure}"
        )


def _get_time_series_output_path(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
    variable: str,
    metric_name: str,
    years: list[int],
    months: list[int],
) -> Path:
    source = source_cfg["source"]

    output_dir = _get_worldclim_feature_output_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_name = build_worldclim_feature_name(
        provider=source["provider"],
        product=source["product"],
        variable=variable,
        metric=metric_name,
        start_year=min(years),
        end_year=max(years),
        start_month=min(months),
        end_month=max(months),
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return output_dir / output_name


def build_worldclim_monthly_time_series_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned features from WorldClim CRU-TS monthly time series.
    """
    _ensure_monthly_time_series(source_cfg)

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = _load_target_grid(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )

    print("[build-time-series] Output AOI:", output_aoi_name)
    print("[build-time-series] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-time-series]")
    print("[build-time-series] Layer structure:", source_cfg["dataset"]["layer_structure"])

    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        scale_factor, resampling, resampling_name = _get_variable_processing(
            source_cfg=source_cfg,
            variable=variable,
            variable_cfg=variable_cfg,
        )

        clipped_dir = _get_worldclim_clipped_base_dir(
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
        print(f"[build-time-series] Variable: {variable}")
        print(f"[build-time-series] Scale factor: {scale_factor}")
        print(f"[build-time-series] Resampling: {resampling_name}")
        print(f"[build-time-series] Clipped dir: {clipped_dir}")

        def spec_factory(year: int, month: int) -> TemporalRasterSpec:
            clipped_name = build_worldclim_clipped_name(
                source_cfg=source_cfg,
                layer_name=variable,
                domain_name=clip_aoi_name,
                year=year,
                month=month,
            )

            return TemporalRasterSpec(
                path=clipped_dir / clipped_name,
                year=year,
                month=month,
                band=1,
            )

        for aggregation_cfg in get_temporal_aggregations(source_cfg):
            if not aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            years = years_from_range(aggregation_cfg["years"])
            months = months_from_range(aggregation_cfg["months"])

            print(
                "[build-time-series] Aggregation:",
                aggregation_cfg.get("name", ""),
                "years=",
                f"{min(years)}-{max(years)}",
                "months=",
                months,
            )

            aggregated, metric_name = aggregate_time_series(
                spec_factory=spec_factory,
                years=years,
                months=months,
                grid=grid,
                resampling=resampling,
                scale_factor=scale_factor,
                aggregation_cfg=aggregation_cfg,
            )

            output_path = _get_time_series_output_path(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                variable=variable,
                metric_name=metric_name,
                years=years,
                months=months,
            )

            metadata = build_feature_metadata(
                source_cfg=source_cfg,
                variable=variable,
                variable_cfg=variable_cfg,
                aggregation_cfg={
                    **aggregation_cfg,
                    "metric": metric_name,
                },
                months=months,
                clip_aoi_name=clip_aoi_name,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                resampling_method_name=resampling_name,
            )

            metadata.update(
                {
                    "years": years,
                    "year_start": min(years),
                    "year_end": max(years),
                    "temporal_axis": "year_month",
                }
            )

            written_path = write_feature_raster(
                output_path=output_path,
                array=aggregated,
                grid=grid,
                metadata=metadata,
                **output_options,
                validate=True,
            )

            print(f"[build-time-series] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths


# =============================================================================
# Future monthly products: CMIP6
# =============================================================================


def _ensure_future_monthly_multiband(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure != "future_monthly_multiband":
        raise ValueError(
            "WorldClim future monthly builder only supports "
            f"layer_structure='future_monthly_multiband', got: {layer_structure}"
        )


def _get_future_clipped_dir(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    file_spec: dict,
) -> Path:
    """
    Return the clipped CMIP6 directory for one variable/GCM/SSP/period.

    Current clipped structure:
      data_interim/sources/worldclim/cmip6_future/clipped/
        <aoi>/<resolution>/<variable>/<gcm>/<ssp>/<period>/
    """
    base_dir = _get_worldclim_clipped_base_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_name=clip_aoi_name,
        variable=file_spec["variable"],
    )

    return (
        base_dir
        / file_spec["gcm"]
        / file_spec["ssp"]
        / file_spec["period"]
    )


def _get_future_month_path(
    clipped_dir: Path,
    source_cfg: dict,
    file_spec: dict,
    clip_aoi_name: str,
    month: int,
) -> Path:
    name = build_worldclim_cmip6_clipped_month_name(
        source_cfg=source_cfg,
        file_spec=file_spec,
        domain_name=clip_aoi_name,
        month=month,
    )

    return clipped_dir / name


def _get_future_output_path(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
    file_spec: dict,
    metric: str,
    months: list[int],
) -> Path:
    source = source_cfg["source"]

    output_dir = _get_worldclim_feature_output_dir(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_name = build_worldclim_cmip6_feature_name(
        provider=source["provider"],
        product=source["product"],
        variable=file_spec["variable"],
        metric=metric,
        gcm=file_spec["gcm"],
        ssp=file_spec["ssp"],
        period=file_spec["period"],
        start_month=min(months),
        end_month=max(months),
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return output_dir / output_name


def _aggregate_future_clipped_months(
    clipped_dir: Path,
    source_cfg: dict,
    file_spec: dict,
    clip_aoi_name: str,
    months: list[int],
    grid,
    resampling,
    scale_factor: float,
    metric: str,
) -> np.ndarray:
    specs = [
        TemporalRasterSpec(
            path=_get_future_month_path(
                clipped_dir=clipped_dir,
                source_cfg=source_cfg,
                file_spec=file_spec,
                clip_aoi_name=clip_aoi_name,
                month=month,
            ),
            month=month,
            band=1,
        )
        for month in months
    ]

    return aggregate_temporal_specs(
        specs=specs,
        grid=grid,
        resampling=resampling,
        scale_factor=scale_factor,
        metric=metric,
    )


def build_worldclim_future_monthly_multiband_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build grid-aligned features from WorldClim CMIP6 future monthly data.

    Raw CMIP6 files are multiband, but the current clipping pipeline produces
    one clipped monthly file per month. This builder consumes those clipped
    monthly files and uses the generic temporal aggregation engine.
    """
    _ensure_future_monthly_multiband(source_cfg)

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = _load_target_grid(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )

    print("[build-future] Output AOI:", output_aoi_name)
    print("[build-future] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-future]")
    print("[build-future] Layer structure:", source_cfg["dataset"]["layer_structure"])

    variables_cfg = source_cfg.get("variables", {})
    file_specs = get_file_specs(source_cfg)
    aggregations = get_temporal_aggregations(source_cfg)

    written_paths: list[Path] = []

    for file_spec in file_specs:
        variable = file_spec["variable"]

        variable_cfg = variables_cfg.get(variable)
        if variable_cfg is None or not variable_cfg.get("enabled", False):
            continue

        scale_factor, resampling, resampling_name = _get_variable_processing(
            source_cfg=source_cfg,
            variable=variable,
            variable_cfg=variable_cfg,
        )

        clipped_dir = _get_future_clipped_dir(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            file_spec=file_spec,
        )

        if not clipped_dir.exists():
            raise FileNotFoundError(
                f"Clipped directory not found for CMIP6 variable '{variable}': "
                f"{clipped_dir}"
            )

        print("==============================")
        print(f"[build-future] Variable: {variable}")
        print(f"[build-future] GCM: {file_spec['gcm']}")
        print(f"[build-future] SSP: {file_spec['ssp']}")
        print(f"[build-future] Period: {file_spec['period']}")
        print(f"[build-future] Scale factor: {scale_factor}")
        print(f"[build-future] Resampling: {resampling_name}")
        print(f"[build-future] Clipped dir: {clipped_dir}")

        for aggregation_cfg in aggregations:
            if not aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            metric = aggregation_cfg["metric"]
            months = months_from_range(aggregation_cfg["months"])

            print(
                f"[build-future] Aggregation: "
                f"{aggregation_cfg.get('name', metric)} "
                f"metric={metric}, months={months}"
            )

            aggregated = _aggregate_future_clipped_months(
                clipped_dir=clipped_dir,
                source_cfg=source_cfg,
                file_spec=file_spec,
                clip_aoi_name=clip_aoi_name,
                months=months,
                grid=grid,
                resampling=resampling,
                scale_factor=scale_factor,
                metric=metric,
            )

            output_path = _get_future_output_path(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                file_spec=file_spec,
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

            metadata.update(
                {
                    "gcm": file_spec["gcm"],
                    "ssp": file_spec["ssp"],
                    "period": file_spec["period"],
                    "temporal_axis": "future_month",
                }
            )

            written_path = write_feature_raster(
                output_path=output_path,
                array=aggregated,
                grid=grid,
                metadata=metadata,
                **output_options,
                validate=True,
            )

            print(f"[build-future] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths


# =============================================================================
# Router
# =============================================================================


def build_worldclim_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Route WorldClim source configs to the correct provider-specific builder.

    Shared raster processing belongs in src.pipeline.
    WorldClim-specific filename/location logic lives here.
    """
    layer_structure = source_cfg["dataset"]["layer_structure"]

    if layer_structure in {"static_index_set", "static_single"}:
        return build_worldclim_static_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "monthly_climatology":
        return build_worldclim_monthly_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "monthly_time_series":
        return build_worldclim_monthly_time_series_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "future_monthly_multiband":
        return build_worldclim_future_monthly_multiband_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    raise NotImplementedError(
        f"Unsupported WorldClim layer_structure: {layer_structure}"
    )