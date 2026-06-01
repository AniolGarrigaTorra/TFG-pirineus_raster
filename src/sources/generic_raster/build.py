from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.io.paths import get_feature_output_dir, get_source_clipped_dir
from src.pipeline.progress import progress_log
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
from src.pipeline.config import (
    get_temporal_aggregations,
    years_from_range,
)
from src.pipeline.temporal import aggregate_stack
from src.sources.generic_raster.naming import (
    build_clipped_name,
    build_feature_name,
    get_enabled_variable_items,
    get_source_resolution,
    validate_generic_raster_source_config,
)


def _get_target_resolution_m(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


def _get_output_options(project_cfg: dict, source_cfg: dict) -> dict[str, Any]:
    output_cfg = source_cfg.get("output", {}) or {}
    return {
        "output_dtype": str(output_cfg.get("dtype", "float32")),
        "nodata": float(project_cfg.get("nodata", -9999.0)),
        "compression": str(output_cfg.get("compression", "LZW")),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _postprocess_array(array: np.ndarray, variable_cfg: dict) -> np.ndarray:
    out = array.astype(np.float32, copy=True)
    valid_range = variable_cfg.get("valid_range")
    if valid_range is not None:
        low = float(valid_range[0])
        high = float(valid_range[1])
        out = np.where((out >= low) & (out <= high), out, np.nan)
    if bool(variable_cfg.get("round_values", False)):
        finite = np.isfinite(out)
        out[finite] = np.rint(out[finite])
    return out.astype(np.float32)


def _reference_year(variable_cfg: dict) -> int | None:
    temporal = variable_cfg.get("temporal", {}) or {}
    if not isinstance(temporal, dict) or temporal.get("reference_year") is None:
        return None
    return int(temporal["reference_year"])


def _yearly_base_variable(variable: str, variable_cfg: dict) -> str:
    return str(variable_cfg.get("generated_from_group") or variable)


def _yearly_aggregation_applies(
    aggregation_cfg: dict,
    base_variable: str,
    variable_items: list[tuple[str, dict]],
) -> bool:
    variables = aggregation_cfg.get("variables")
    if variables is None:
        return True
    selected = {str(item) for item in variables}
    return base_variable in selected or any(variable in selected for variable, _ in variable_items)


def _yearly_clipped_path(
    *,
    project_cfg: dict,
    source_cfg: dict,
    variable: str,
    clip_aoi_name: str,
) -> Path:
    source = source_cfg["source"]
    clipped_dir = get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=clip_aoi_name,
        source_resolution=get_source_resolution(source_cfg),
        variable=variable,
    )
    return clipped_dir / build_clipped_name(source_cfg, variable, clip_aoi_name)


def _build_yearly_static_aggregations(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    variables_by_base: dict[str, list[tuple[str, dict]]] = {}
    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        if _reference_year(variable_cfg) is None:
            continue
        base_variable = _yearly_base_variable(variable, variable_cfg)
        variables_by_base.setdefault(base_variable, []).append((variable, variable_cfg))

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    written_paths: list[Path] = []

    progress_log("[build-yearly] Temporal output mode: aggregate")
    print_grid_context(grid, prefix="[build-yearly]")

    for aggregation_cfg in get_temporal_aggregations(source_cfg):
        metric = aggregation_cfg["metric"]
        years = years_from_range(aggregation_cfg["years"])
        year_set = set(years)
        aggregation_name = str(aggregation_cfg.get("name", metric))

        for base_variable, variable_items in variables_by_base.items():
            if not _yearly_aggregation_applies(aggregation_cfg, base_variable, variable_items):
                continue

            selected_items = [
                (variable, variable_cfg)
                for variable, variable_cfg in variable_items
                if _reference_year(variable_cfg) in year_set
            ]
            selected_items.sort(key=lambda item: _reference_year(item[1]) or 0)

            if not selected_items:
                continue

            arrays: list[np.ndarray] = []
            first_variable, first_cfg = selected_items[0]
            resampling = get_variable_resampling_method(source_cfg, first_variable)
            resampling_name = get_variable_resampling_method_name(source_cfg, first_variable)

            progress_log(f"[build-yearly] Variable: {base_variable}")
            progress_log(f"[build-yearly] Aggregation: {aggregation_name}")
            progress_log(f"[build-yearly] Years: {years[0]}-{years[-1]}")

            for variable, variable_cfg in selected_items:
                clipped_path = _yearly_clipped_path(
                    project_cfg=project_cfg,
                    source_cfg=source_cfg,
                    variable=variable,
                    clip_aoi_name=clip_aoi_name,
                )
                if not clipped_path.exists():
                    if not bool(variable_cfg.get("required", True)):
                        progress_log(f"[build-yearly] Optional clipped raster missing, skipping: {clipped_path}")
                        continue
                    raise FileNotFoundError(
                        f"Missing clipped raster: {clipped_path}\nRun the clip stage first."
                    )

                array = read_raster_to_grid(
                    raster_path=clipped_path,
                    grid=grid,
                    resampling=resampling,
                    band=int(variable_cfg.get("band", 1)),
                    scale_factor=float(variable_cfg.get("scale_factor", 1.0)),
                    resampling_method_name=resampling_name,
                )
                arrays.append(_postprocess_array(array, variable_cfg))

            if not arrays:
                continue

            aggregated = aggregate_stack(
                stack=np.stack(arrays, axis=0),
                metric=metric,
            ).astype(np.float32)

            output_variable = f"{base_variable}_{aggregation_name}"
            output_path = output_dir / build_feature_name(
                source_cfg=source_cfg,
                variable=output_variable,
                domain_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
            )

            metadata = build_feature_metadata(
                source_cfg=source_cfg,
                variable=base_variable,
                variable_cfg=first_cfg,
                aggregation_cfg=aggregation_cfg,
                months=[],
                clip_aoi_name=clip_aoi_name,
                output_aoi_name=output_aoi_name,
                target_resolution_m=target_resolution_m,
                resampling_method_name=resampling_name,
            )
            metadata.update(
                {
                    "years": years,
                    "year_start": years[0],
                    "year_end": years[-1],
                    "temporal_output_mode": "aggregate",
                    "input_variables": [item[0] for item in selected_items],
                }
            )

            written_paths.append(
                write_feature_raster(
                    output_path=output_path,
                    array=aggregated,
                    grid=grid,
                    metadata={key: value for key, value in metadata.items() if value is not None},
                    **output_options,
                    validate=True,
                )
            )

    return written_paths


def build_generic_raster_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
    provider: str | None = None,
) -> list[Path]:
    validate_generic_raster_source_config(source_cfg, provider=provider)

    if (
        source_cfg.get("dataset", {}).get("layer_structure") == "yearly_static_collection"
        and source_cfg.get("temporal", {}).get("output_mode") == "aggregate"
    ):
        return _build_yearly_static_aggregations(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    source = source_cfg["source"]
    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    progress_log(f"[build] Provider: {source['provider']}")
    progress_log(f"[build] Product: {source['product']}")
    progress_log(f"[build] Output AOI: {output_aoi_name}")
    progress_log(f"[build] Clip AOI: {clip_aoi_name}")
    progress_log(f"[build] Target resolution: {target_resolution_m}")
    print_grid_context(grid, prefix="[build]")

    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))
        resampling = get_variable_resampling_method(source_cfg, variable)
        resampling_name = get_variable_resampling_method_name(source_cfg, variable)

        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
            domain_name=clip_aoi_name,
            source_resolution=get_source_resolution(source_cfg),
            variable=variable,
        )
        clipped_path = clipped_dir / build_clipped_name(source_cfg, variable, clip_aoi_name)

        if not clipped_path.exists():
            if not bool(variable_cfg.get("required", True)):
                progress_log(f"[build] Optional clipped raster missing, skipping: {clipped_path}")
                continue
            raise FileNotFoundError(
                f"Missing clipped raster: {clipped_path}\nRun the clip stage first."
            )

        output_dir = get_feature_output_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
            domain_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
        )
        output_path = output_dir / build_feature_name(
            source_cfg=source_cfg,
            variable=variable,
            domain_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
        )

        progress_log(f"[build] Variable: {variable}")
        progress_log(f"[build] Description: {variable_cfg.get('description', '')}")
        progress_log(f"[build] Clipped path: {clipped_path}")
        progress_log(f"[build] Output path: {output_path}")

        grid_array = read_raster_to_grid(
            raster_path=clipped_path,
            grid=grid,
            resampling=resampling,
            band=int(variable_cfg.get("band", 1)),
            scale_factor=scale_factor,
            resampling_method_name=resampling_name,
        )
        grid_array = _postprocess_array(grid_array, variable_cfg)

        metadata = build_static_feature_metadata(
            source_cfg=source_cfg,
            layer_name=variable,
            layer_cfg=variable_cfg,
            clip_aoi_name=clip_aoi_name,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            resampling_method_name=resampling_name,
        )
        metadata.update(
            {
                "data_type": variable_cfg.get("data_type")
                or source_cfg.get("dataset", {}).get("data_type"),
                "native_resolution_m": variable_cfg.get("native_resolution_m")
                or source_cfg.get("dataset", {}).get("native_resolution_m"),
                "reference_year": (
                    variable_cfg.get("temporal", {}).get("reference_year")
                    if isinstance(variable_cfg.get("temporal"), dict)
                    else source.get("source_period")
                ),
            }
        )

        written_paths.append(
            write_feature_raster(
                output_path=output_path,
                array=grid_array,
                grid=grid,
                metadata={key: value for key, value in metadata.items() if value is not None},
                **output_options,
                validate=True,
            )
        )

    return written_paths
