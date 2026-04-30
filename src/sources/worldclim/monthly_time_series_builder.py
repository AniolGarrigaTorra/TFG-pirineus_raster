from __future__ import annotations

from pathlib import Path

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_clipped_dir
from src.pipeline.aggregation import months_from_range
from src.pipeline.feature_writer import write_feature_raster
from src.pipeline.grid_context import load_grid_context, print_grid_context
from src.pipeline.metadata import build_feature_metadata
from src.pipeline.resampling import (
    get_variable_resampling_method,
    get_variable_resampling_method_name,
)
from src.pipeline.source_config import (
    aggregation_applies_to_variable,
    get_enabled_variable_items,
    get_temporal_aggregations,
    years_from_range,
)
from src.pipeline.temporal_processing import (
    TemporalRasterSpec,
    aggregate_time_series,
)
from src.sources.worldclim.naming import (
    build_worldclim_clipped_name,
    build_worldclim_feature_name,
)


def _ensure_monthly_time_series(source_cfg: dict) -> None:
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure != "monthly_time_series":
        raise ValueError(
            "WorldClim monthly time series builder only supports "
            f"layer_structure='monthly_time_series', got: {layer_structure}"
        )


def _get_time_series_clipped_dir(
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

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

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

    The provider-specific part is only:
      - locating year-month clipped rasters
      - naming final outputs

    Temporal aggregation itself is handled by src.pipeline.temporal_processing.
    """
    _ensure_monthly_time_series(source_cfg)

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

    print("[build-time-series] Output AOI:", output_aoi_name)
    print("[build-time-series] Clip AOI:", clip_aoi_name)
    print_grid_context(grid, prefix="[build-time-series]")
    print("[build-time-series] Layer structure:", source_cfg["dataset"]["layer_structure"])

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

        clipped_dir = _get_time_series_clipped_dir(
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
                output_dtype=output_dtype,
                nodata=nodata,
                compression=compression,
                write_sidecar=write_sidecar,
                validate=True,
            )

            print(f"[build-time-series] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths