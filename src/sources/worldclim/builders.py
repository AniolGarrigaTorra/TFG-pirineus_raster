from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject

from src.sources.worldclim.static_builder import build_worldclim_static_features

from src.io.paths import (
    ensure_dir,
    get_grid_path,
    get_source_clipped_dir,
    get_feature_output_dir,
)
from src.pipeline.aggregation import aggregate_stack, months_from_range
from src.pipeline.metadata import (
    build_feature_metadata,
    metadata_to_geotiff_tags,
    write_sidecar_json,
)
from src.pipeline.resampling import (
    get_resampling_method,
    get_variable_resampling_method,
    get_variable_resampling_method_name,
)
from src.pipeline.validation import validate_raster_matches_grid
from src.pipeline.source_config import (
    aggregation_applies_to_variable,
    get_enabled_index_items,
    get_enabled_variable_items,
    get_enabled_variables,
    get_static_layer_items,
    get_temporal_aggregations,
    get_time_series_metric_name,
    years_from_range,
)
from src.sources.worldclim.naming import (
    build_worldclim_clipped_name,
    build_worldclim_feature_name,
    build_worldclim_cmip6_clipped_month_name,
    build_worldclim_cmip6_feature_name,
    get_file_specs,
)


def _aggregation_applies_to_variable(
    aggregation_cfg: dict,
    variable: str,
) -> bool:
    """
    If aggregation config contains a 'variables' list, use it as a filter.
    Otherwise, apply aggregation to all enabled variables.
    """
    variables = aggregation_cfg.get("variables")

    if variables is None:
        return True

    return variable in variables


def _read_and_reproject_raster_to_grid(
    source_path: Path,
    grid_profile: dict,
    grid_transform,
    grid_crs,
    grid_height: int,
    grid_width: int,
    resampling,
    source_nodata,
) -> np.ndarray:
    """
    Read one clipped monthly raster and reproject/resample it exactly to the target grid.
    """
    destination = np.full(
        shape=(grid_height, grid_width),
        fill_value=np.nan,
        dtype=np.float32,
    )

    with rasterio.open(source_path) as src:
        source_data = src.read(1).astype(np.float32)

        src_nodata = src.nodata
        if src_nodata is not None:
            source_data[source_data == src_nodata] = np.nan

        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid_transform,
            dst_crs=grid_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    return destination

def _get_enabled_variables(source_cfg: dict) -> list[str]:
    return get_enabled_variables(source_cfg)


def _get_enabled_indices(source_cfg: dict) -> list[tuple[str, dict]]:
    return get_enabled_index_items(source_cfg)


def _get_enabled_static_variables(source_cfg: dict) -> list[tuple[str, dict]]:
    return get_enabled_variable_items(source_cfg)


def _years_from_range(year_range: list[int]) -> list[int]:
    return years_from_range(year_range)


def _get_time_series_metric_name(aggregation_cfg: dict) -> str:
    return get_time_series_metric_name(aggregation_cfg)


def _get_static_resampling_name(source_cfg: dict, layer_name: str) -> str:
    resampling_cfg = source_cfg.get("resampling", {})
    default_method = resampling_cfg.get("default", "nearest")
    by_variable = resampling_cfg.get("by_variable", {})
    return by_variable.get(layer_name, default_method)


def build_worldclim_future_monthly_multiband_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {})

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]
    target_resolution_m = int(processing["target_resolution_m"])

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]

    nodata = float(project_cfg.get("nodata", -9999.0))
    output_dtype = output_cfg.get("dtype", "float32")
    compression = output_cfg.get("compression", "LZW")
    write_sidecar = bool(output_cfg.get("write_sidecar_json", True))

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Target grid does not exist: {grid_path}\n"
            f"Create it first with make_grid.py for AOI={output_aoi_name}, "
            f"resolution={target_resolution_m}m."
        )

    with rasterio.open(grid_path) as grid:
        grid_profile = grid.profile.copy()
        grid_transform = grid.transform
        grid_crs = grid.crs
        grid_height = grid.height
        grid_width = grid.width

    file_specs = get_file_specs(source_cfg)
    aggregations = source_cfg.get("temporal_aggregations", [])

    if not aggregations:
        raise ValueError("No temporal_aggregations found in CMIP6 source config.")

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

    written_paths: list[Path] = []

    for file_spec in file_specs:
        variable = file_spec["variable"]
        variable_cfg = source_cfg["variables"][variable]
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(source_cfg, variable)
        resampling_name = get_variable_resampling_method_name(source_cfg, variable)

        clipped_dir = (
            get_source_clipped_dir(
                project_cfg=project_cfg,
                provider=provider,
                product=product,
                domain_name=clip_aoi_name,
                source_resolution=source_resolution,
                variable=variable,
            )
            / file_spec["gcm"]
            / file_spec["ssp"]
            / file_spec["period"]
        )

        print("==============================")
        print(
            f"[build-cmip6] Variable={variable}, GCM={file_spec['gcm']}, "
            f"SSP={file_spec['ssp']}, period={file_spec['period']}"
        )
        print(f"[build-cmip6] Clipped dir: {clipped_dir}")

        for aggregation_cfg in aggregations:
            if not _aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            metric = aggregation_cfg["metric"]
            months = months_from_range(aggregation_cfg["months"])

            arrays = []

            for month in months:
                clipped_name = build_worldclim_cmip6_clipped_month_name(
                    source_cfg=source_cfg,
                    file_spec=file_spec,
                    domain_name=clip_aoi_name,
                    month=month,
                )

                clipped_path = clipped_dir / clipped_name

                if not clipped_path.exists():
                    raise FileNotFoundError(
                        f"Missing clipped CMIP6 monthly raster: {clipped_path}"
                    )

                grid_array = _read_and_reproject_raster_to_grid(
                    source_path=clipped_path,
                    grid_profile=grid_profile,
                    grid_transform=grid_transform,
                    grid_crs=grid_crs,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    resampling=resampling,
                    source_nodata=None,
                )

                grid_array = grid_array * scale_factor
                arrays.append(grid_array)

            stack = np.stack(arrays, axis=0)
            aggregated = aggregate_stack(stack, metric=metric).astype(np.float32)

            output_array = np.where(
                np.isfinite(aggregated),
                aggregated,
                nodata,
            ).astype(output_dtype)

            output_name = build_worldclim_cmip6_feature_name(
                provider=provider,
                product=product,
                variable=variable,
                metric=metric,
                gcm=file_spec["gcm"],
                ssp=file_spec["ssp"],
                period=file_spec["period"],
                start_month=min(months),
                end_month=max(months),
                domain_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
            )

            output_path = output_dir / output_name

            output_profile = grid_profile.copy()
            output_profile.update(
                {
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": output_dtype,
                    "nodata": nodata,
                    "compress": compression,
                }
            )

            for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
                output_profile.pop(key, None)

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
                    "future_period": file_spec["period"],
                    "worldclim_variable_code": file_spec["worldclim_code"],
                    "dataset_layer_structure": "future_monthly_multiband",
                }
            )

            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(output_array, 1)
                dst.update_tags(**metadata_to_geotiff_tags(metadata))

            if write_sidecar:
                json_path = write_sidecar_json(metadata, output_path)
                print(f"[build-cmip6] Metadata JSON: {json_path}")

            validate_raster_matches_grid(
                raster_path=output_path,
                grid_path=grid_path,
            )

            print(f"[build-cmip6] Written: {output_path}")
            written_paths.append(output_path)

    return written_paths

def build_worldclim_monthly_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned feature TIFFs from clipped WorldClim monthly rasters.

    The clipped rasters remain in WorldClim native CRS/resolution.
    This stage reprojects/resamples them to the project grid, applies scale_factor,
    aggregates months, and writes final feature TIFFs.
    """
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {})

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]
    target_resolution_m = int(processing["target_resolution_m"])

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]

    nodata = float(project_cfg.get("nodata", -9999.0))
    output_dtype = output_cfg.get("dtype", "float32")
    compression = output_cfg.get("compression", "LZW")
    write_sidecar = bool(output_cfg.get("write_sidecar_json", True))

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Target grid does not exist: {grid_path}\n"
            f"Create it first with make_grid.py for AOI={output_aoi_name}, "
            f"resolution={target_resolution_m}m."
        )

    with rasterio.open(grid_path) as grid:
        grid_profile = grid.profile.copy()
        grid_transform = grid.transform
        grid_crs = grid.crs
        grid_height = grid.height
        grid_width = grid.width

    print("[build] Output AOI:", output_aoi_name)
    print("[build] Clip AOI:", clip_aoi_name)
    print("[build] Grid:", grid_path)
    print("[build] Grid CRS:", grid_crs)
    print("[build] Grid shape:", grid_width, grid_height)
    print("[build] Target resolution:", target_resolution_m)

    enabled_variables = _get_enabled_variables(source_cfg)
    aggregations = source_cfg.get("temporal_aggregations", [])

    if not aggregations:
        raise ValueError("No temporal_aggregations found in source config.")

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

    written_paths: list[Path] = []

    for variable in enabled_variables:
        variable_cfg = source_cfg["variables"][variable]
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(source_cfg, variable)
        resampling_name = get_variable_resampling_method_name(source_cfg, variable)

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=variable,
        )

        if not clipped_dir.exists():
            raise FileNotFoundError(
                f"Clipped directory not found for variable '{variable}': {clipped_dir}"
            )

        print("==============================")
        print(f"[build] Variable: {variable}")
        print(f"[build] Scale factor: {scale_factor}")
        print(f"[build] Resampling: {resampling_name}")
        print(f"[build] Clipped dir: {clipped_dir}")

        for aggregation_cfg in aggregations:
            if not _aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            metric = aggregation_cfg["metric"]
            months = months_from_range(aggregation_cfg["months"])

            print(
                f"[build] Aggregation: {aggregation_cfg.get('name', metric)} "
                f"metric={metric}, months={months}"
            )

            monthly_arrays = []

            for month in months:
                clipped_name = build_worldclim_clipped_name(
                    source_cfg=source_cfg,
                    layer_name=variable,
                    domain_name=clip_aoi_name,
                    month=month,
                )

                clipped_path = clipped_dir / clipped_name

                if not clipped_path.exists():
                    raise FileNotFoundError(
                        f"Missing clipped monthly raster: {clipped_path}"
                    )

                monthly_grid = _read_and_reproject_raster_to_grid(
                    source_path=clipped_path,
                    grid_profile=grid_profile,
                    grid_transform=grid_transform,
                    grid_crs=grid_crs,
                    grid_height=grid_height,
                    grid_width=grid_width,
                    resampling=resampling,
                    source_nodata=None,
                )

                # Apply WorldClim scale factor after reprojection.
                monthly_grid = monthly_grid * scale_factor

                monthly_arrays.append(monthly_grid)

            stack = np.stack(monthly_arrays, axis=0)
            aggregated = aggregate_stack(stack, metric=metric).astype(np.float32)

            # Convert NaN to project nodata.
            aggregated_out = np.where(
                np.isfinite(aggregated),
                aggregated,
                nodata,
            ).astype(output_dtype)

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

            output_path = output_dir / output_name

            output_profile = grid_profile.copy()
            output_profile.update(
                {
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": output_dtype,
                    "nodata": nodata,
                    "compress": compression,
                }
            )

            # Avoid carrying over incompatible source/grid block settings.
            for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
                output_profile.pop(key, None)

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

            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(aggregated_out, 1)
                dst.update_tags(**metadata_to_geotiff_tags(metadata))

            if write_sidecar:
                json_path = write_sidecar_json(metadata, output_path)
                print(f"[build] Metadata JSON: {json_path}")

            validate_raster_matches_grid(
                raster_path=output_path,
                grid_path=grid_path,
            )

            print(f"[build] Written: {output_path}")
            written_paths.append(output_path)

    return written_paths


def build_worldclim_monthly_time_series_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned features from WorldClim CRU-TS monthly time series.

    Supports aggregations across selected years and months.
    """
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    output_cfg = source_cfg.get("output", {})

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]
    target_resolution_m = int(processing["target_resolution_m"])

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]

    nodata = float(project_cfg.get("nodata", -9999.0))
    output_dtype = output_cfg.get("dtype", "float32")
    compression = output_cfg.get("compression", "LZW")
    write_sidecar = bool(output_cfg.get("write_sidecar_json", True))

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Target grid does not exist: {grid_path}\n"
            f"Create it first with make_grid.py for AOI={output_aoi_name}, "
            f"resolution={target_resolution_m}m."
        )

    with rasterio.open(grid_path) as grid:
        grid_profile = grid.profile.copy()
        grid_transform = grid.transform
        grid_crs = grid.crs
        grid_height = grid.height
        grid_width = grid.width

    print("[build-timeseries] Output AOI:", output_aoi_name)
    print("[build-timeseries] Clip AOI:", clip_aoi_name)
    print("[build-timeseries] Grid:", grid_path)
    print("[build-timeseries] Target resolution:", target_resolution_m)

    enabled_variables = _get_enabled_variables(source_cfg)
    aggregations = source_cfg.get("temporal_aggregations", [])

    if not aggregations:
        raise ValueError("No temporal_aggregations found in source config.")

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

    written_paths: list[Path] = []

    for variable in enabled_variables:
        variable_cfg = source_cfg["variables"][variable]
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(source_cfg, variable)
        resampling_name = get_variable_resampling_method_name(source_cfg, variable)

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=variable,
        )

        if not clipped_dir.exists():
            raise FileNotFoundError(
                f"Clipped directory not found for variable '{variable}': {clipped_dir}"
            )

        print("==============================")
        print(f"[build-timeseries] Variable: {variable}")
        print(f"[build-timeseries] Scale factor: {scale_factor}")
        print(f"[build-timeseries] Resampling: {resampling_name}")
        print(f"[build-timeseries] Clipped dir: {clipped_dir}")

        for aggregation_cfg in aggregations:
            if not _aggregation_applies_to_variable(aggregation_cfg, variable):
                continue

            years = _years_from_range(aggregation_cfg["years"])
            months = months_from_range(aggregation_cfg["months"])

            print(
                f"[build-timeseries] Aggregation: {aggregation_cfg.get('name', '')} "
                f"years={years[0]}-{years[-1]}, months={months}"
            )

            # Case A: simple aggregation across all selected year-month rasters.
            if "metric" in aggregation_cfg:
                arrays = []

                for year in years:
                    for month in months:
                        clipped_name = build_worldclim_clipped_name(
                            source_cfg=source_cfg,
                            layer_name=variable,
                            domain_name=clip_aoi_name,
                            year=year,
                            month=month,
                        )

                        clipped_path = clipped_dir / clipped_name

                        if not clipped_path.exists():
                            raise FileNotFoundError(
                                f"Missing clipped monthly raster: {clipped_path}"
                            )

                        grid_array = _read_and_reproject_raster_to_grid(
                            source_path=clipped_path,
                            grid_profile=grid_profile,
                            grid_transform=grid_transform,
                            grid_crs=grid_crs,
                            grid_height=grid_height,
                            grid_width=grid_width,
                            resampling=resampling,
                            source_nodata=None,
                        )

                        grid_array = grid_array * scale_factor
                        arrays.append(grid_array)

                stack = np.stack(arrays, axis=0)
                metric_name = aggregation_cfg["metric"]
                aggregated = aggregate_stack(stack, metric=metric_name).astype(np.float32)

            # Case B: aggregate within each year, then aggregate across years.
            else:
                within_year_metric = aggregation_cfg["within_year_metric"]
                across_year_metric = aggregation_cfg["across_year_metric"]
                metric_name = _get_time_series_metric_name(aggregation_cfg)

                yearly_arrays = []

                for year in years:
                    monthly_arrays = []

                    for month in months:
                        clipped_name = build_worldclim_clipped_name(
                            source_cfg=source_cfg,
                            layer_name=variable,
                            domain_name=clip_aoi_name,
                            year=year,
                            month=month,
                        )

                        clipped_path = clipped_dir / clipped_name

                        if not clipped_path.exists():
                            raise FileNotFoundError(
                                f"Missing clipped monthly raster: {clipped_path}"
                            )

                        grid_array = _read_and_reproject_raster_to_grid(
                            source_path=clipped_path,
                            grid_profile=grid_profile,
                            grid_transform=grid_transform,
                            grid_crs=grid_crs,
                            grid_height=grid_height,
                            grid_width=grid_width,
                            resampling=resampling,
                            source_nodata=None,
                        )

                        grid_array = grid_array * scale_factor
                        monthly_arrays.append(grid_array)

                    year_stack = np.stack(monthly_arrays, axis=0)
                    year_aggregated = aggregate_stack(
                        year_stack,
                        metric=within_year_metric,
                    ).astype(np.float32)

                    yearly_arrays.append(year_aggregated)

                yearly_stack = np.stack(yearly_arrays, axis=0)
                aggregated = aggregate_stack(
                    yearly_stack,
                    metric=across_year_metric,
                ).astype(np.float32)

            aggregated_out = np.where(
                np.isfinite(aggregated),
                aggregated,
                nodata,
            ).astype(output_dtype)

            output_name = build_worldclim_feature_name(
                provider=provider,
                product=product,
                variable=variable,
                metric=metric_name,
                start_year=min(years),
                end_year=max(years),
                start_month=min(months),
                end_month=max(months),
                domain_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
            )

            output_path = output_dir / output_name

            output_profile = grid_profile.copy()
            output_profile.update(
                {
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": output_dtype,
                    "nodata": nodata,
                    "compress": compression,
                }
            )

            for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
                output_profile.pop(key, None)

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

            metadata["years"] = years
            metadata["year_start"] = min(years)
            metadata["year_end"] = max(years)
            metadata["time_series_aggregation"] = aggregation_cfg

            with rasterio.open(output_path, "w", **output_profile) as dst:
                dst.write(aggregated_out, 1)
                dst.update_tags(**metadata_to_geotiff_tags(metadata))

            if write_sidecar:
                json_path = write_sidecar_json(metadata, output_path)
                print(f"[build-timeseries] Metadata JSON: {json_path}")

            validate_raster_matches_grid(
                raster_path=output_path,
                grid_path=grid_path,
            )

            print(f"[build-timeseries] Written: {output_path}")
            written_paths.append(output_path)

    return written_paths


def build_worldclim_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Route WorldClim feature building according to dataset.layer_structure.
    """
    layer_structure = source_cfg.get("dataset", {}).get(
        "layer_structure",
        "monthly_climatology",
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

    if layer_structure in {"static_index_set", "static_single"}:
        return build_worldclim_static_features(
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
        f"Unsupported layer_structure for build stage: {layer_structure}"
    )