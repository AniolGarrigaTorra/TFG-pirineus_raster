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
    get_temporal_aggregations,
)
from src.pipeline.temporal_processing import aggregate_monthly_bands
from src.sources.worldclim.naming import (
    build_worldclim_cmip6_clipped_month_name,
    build_worldclim_cmip6_feature_name,
    get_file_specs,
)


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
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    base_dir = get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=clip_aoi_name,
        source_resolution=processing["source_resolution"],
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
    """
    Current clipping pipeline writes one clipped GeoTIFF per month.

    Even if raw CMIP6 is multiband, after clipping we use the existing
    clipped monthly files produced by WorldClim clipping logic.
    """
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

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

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


def _aggregate_clipped_month_files(
    clipped_dir: Path,
    source_cfg: dict,
    file_spec: dict,
    clip_aoi_name: str,
    months: list[int],
    grid,
    resampling,
    scale_factor: float,
    metric: str,
):
    """
    Aggregate CMIP6 clipped monthly files.

    This uses the same generic temporal stack mechanism as other monthly data,
    but the file locator is CMIP6-specific.
    """
    from src.pipeline.temporal_processing import TemporalRasterSpec, aggregate_temporal_specs

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

        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(
            source_cfg=source_cfg,
            variable=variable,
        )
        resampling_name = get_variable_resampling_method_name(
            source_cfg=source_cfg,
            variable=variable,
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

            aggregated = _aggregate_clipped_month_files(
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
                output_dtype=output_dtype,
                nodata=nodata,
                compression=compression,
                write_sidecar=write_sidecar,
                validate=True,
            )

            print(f"[build-future] Written: {written_path}")
            written_paths.append(written_path)

    return written_paths