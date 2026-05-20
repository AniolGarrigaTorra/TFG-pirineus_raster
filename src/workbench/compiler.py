from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.io.config import load_yaml, resolve_path
from src.pipeline.config import validate_run_config
from src.pipeline.variable_expansion import expand_source_config
from src.workbench.catalog import SUPPORTED_METRICS, SUPPORTED_RESAMPLING


class ConfigValidationError(ValueError):
    """Raised when a researcher-facing run config is not valid."""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _sanitize_token(value: Any) -> str:
    text = str(value)
    for old, new in [
        ("-", "_"),
        (" ", "_"),
        ("/", "_"),
        (":", "_"),
        (".", "_"),
    ]:
        text = text.replace(old, new)
    return text


def _disable_all(collection: dict[str, Any]) -> None:
    for cfg in collection.values():
        if isinstance(cfg, dict):
            cfg["enabled"] = False


def _enable_selected(
    collection: dict[str, Any],
    selected: list[str],
    context: str,
) -> None:
    unknown = sorted(set(selected) - set(collection))
    if unknown:
        raise ConfigValidationError(
            f"Unknown {context}: {unknown}. Available: {sorted(collection)}"
        )

    _disable_all(collection)
    for name in selected:
        collection[name]["enabled"] = True


def _compile_variable_selection(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    selected_variables = [str(item) for item in _as_list(select_cfg.get("variables"))]
    selected_indices = [str(item) for item in _as_list(select_cfg.get("indices"))]

    variables = cfg.get("variables", {}) or {}
    indices = cfg.get("indices", {}) or {}

    if selected_variables:
        if variables:
            _enable_selected(variables, selected_variables, "variables")
        elif indices:
            _enable_selected(indices, selected_variables, "indices")
        else:
            raise ConfigValidationError(
                "This source does not expose variables or indices."
            )

    if selected_indices:
        if not indices:
            raise ConfigValidationError("This source does not expose indices.")
        _enable_selected(indices, selected_indices, "indices")


def _compile_layer_selection(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    selected_layers = [str(item) for item in _as_list(select_cfg.get("layers"))]
    if not selected_layers:
        return

    datasets = cfg.get("datasets", {}) or {}
    if not datasets:
        raise ConfigValidationError("This source does not expose vector layers.")

    available: dict[str, tuple[str, str]] = {}
    for dataset_name, dataset_cfg in datasets.items():
        for layer_name in (dataset_cfg.get("layers", {}) or {}):
            available[f"{dataset_name}.{layer_name}"] = (dataset_name, layer_name)
            available.setdefault(layer_name, (dataset_name, layer_name))

    unknown = sorted(set(selected_layers) - set(available))
    if unknown:
        raise ConfigValidationError(
            f"Unknown layers: {unknown}. Available: {sorted(available)}"
        )

    for dataset_cfg in datasets.values():
        dataset_cfg["enabled"] = False
        for layer_cfg in (dataset_cfg.get("layers", {}) or {}).values():
            layer_cfg["enabled"] = False

    for selected in selected_layers:
        dataset_name, layer_name = available[selected]
        datasets[dataset_name]["enabled"] = True
        datasets[dataset_name]["layers"][layer_name]["enabled"] = True


def _compile_dimensions(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    dimensions_cfg = select_cfg.get("dimensions", {}) or {}

    for key, selected_values in dimensions_cfg.items():
        selected = [str(item) for item in _as_list(selected_values)]
        if not selected:
            continue

        available = cfg.get(key)
        if available is None:
            raise ConfigValidationError(
                f"Source does not expose dimension {key!r}."
            )

        available_values = [str(item) for item in available]
        unknown = sorted(set(selected) - set(available_values))
        if unknown:
            raise ConfigValidationError(
                f"Unknown values for dimension {key!r}: {unknown}. "
                f"Available: {available_values}"
            )

        cfg[key] = selected


def _aggregation_variable_names(cfg: dict[str, Any]) -> set[str]:
    return set(cfg.get("variables", {}) or {}) | set(cfg.get("indices", {}) or {})


def _validate_aggregation(
    aggregation: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    if "name" not in aggregation:
        raise ConfigValidationError(f"Aggregation is missing name: {aggregation}")

    metric = aggregation.get("metric")
    within = aggregation.get("within_year_metric")
    across = aggregation.get("across_year_metric")

    for value in [metric, within, across]:
        if value is not None and value not in SUPPORTED_METRICS:
            raise ConfigValidationError(
                f"Unsupported aggregation metric {value!r}. "
                f"Supported: {SUPPORTED_METRICS}"
            )

    if metric is None and not (within and across):
        raise ConfigValidationError(
            f"Aggregation {aggregation['name']!r} must define either metric "
            "or within_year_metric + across_year_metric."
        )

    known_variables = _aggregation_variable_names(cfg)
    selected_variables = [str(item) for item in aggregation.get("variables", [])]
    unknown = sorted(set(selected_variables) - known_variables)
    if unknown:
        raise ConfigValidationError(
            f"Aggregation {aggregation['name']!r} references unknown variables: {unknown}"
        )


def _compile_aggregations(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    aggregation_select = select_cfg.get("aggregations")
    if aggregation_select is None:
        return

    if isinstance(aggregation_select, list):
        use = [str(item) for item in aggregation_select]
        custom: list[dict[str, Any]] = []
    elif isinstance(aggregation_select, dict):
        use = [str(item) for item in _as_list(aggregation_select.get("use"))]
        custom = list(aggregation_select.get("custom", []) or [])
    else:
        raise ConfigValidationError("select.aggregations must be a list or dictionary.")

    existing = cfg.get("temporal_aggregations", []) or []
    by_name = {item.get("name"): item for item in existing if isinstance(item, dict)}

    unknown = sorted(set(use) - set(by_name))
    if unknown:
        raise ConfigValidationError(
            f"Unknown aggregation presets: {unknown}. Available: {sorted(by_name)}"
        )

    selected = [deepcopy(by_name[name]) for name in use]
    selected.extend(deepcopy(custom))

    for aggregation in selected:
        _validate_aggregation(aggregation, cfg)

    cfg["temporal_aggregations"] = selected


def _compile_resampling_overrides(
    cfg: dict[str, Any],
    source_entry: dict[str, Any],
) -> None:
    overrides = (source_entry.get("overrides", {}) or {}).get("resampling", {}) or {}
    if not overrides:
        return

    resampling = cfg.setdefault("resampling", {})

    default = overrides.get("default")
    if default is not None:
        if default not in SUPPORTED_RESAMPLING:
            raise ConfigValidationError(f"Unsupported resampling method: {default}")
        resampling["default"] = default

    by_variable = overrides.get("by_variable") or overrides.get("variables") or {}
    if by_variable:
        target = resampling.setdefault("by_variable", {})
        for variable, method in by_variable.items():
            if method not in SUPPORTED_RESAMPLING:
                raise ConfigValidationError(
                    f"Unsupported resampling method for {variable}: {method}"
                )
            target[variable] = method


def _compile_processing_overrides(
    cfg: dict[str, Any],
    source_entry: dict[str, Any],
) -> None:
    overrides = source_entry.get("overrides", {}) or {}
    processing_overrides = overrides.get("processing", {}) or {}
    if not processing_overrides:
        return

    processing = cfg.setdefault("processing", {})

    if processing_overrides.get("source_resolution") is not None:
        processing["source_resolution"] = str(processing_overrides["source_resolution"])

    if processing_overrides.get("target_resolution_m") is not None:
        processing["target_resolution_m"] = int(processing_overrides["target_resolution_m"])


def _compile_download_overrides(
    cfg: dict[str, Any],
    source_entry: dict[str, Any],
) -> None:
    overrides = source_entry.get("overrides", {}) or {}
    download_overrides = overrides.get("download", {}) or {}
    if not download_overrides:
        return

    download = cfg.setdefault("download", {})

    if "keep_raw_after_clip" in download_overrides:
        keep_raw = bool(download_overrides["keep_raw_after_clip"])
        download["keep_raw_after_clip"] = keep_raw
        download["delete_raw_after_clip"] = not keep_raw
        download["keep_global_file_after_clip"] = keep_raw
        download["keep_global_zip_after_clip"] = keep_raw
        download["keep_raw_zip_after_clip"] = keep_raw

    if "delete_raw_after_clip" in download_overrides:
        delete_raw = bool(download_overrides["delete_raw_after_clip"])
        download["delete_raw_after_clip"] = delete_raw
        download["keep_raw_after_clip"] = not delete_raw
        download["keep_global_file_after_clip"] = not delete_raw
        download["keep_global_zip_after_clip"] = not delete_raw
        download["keep_raw_zip_after_clip"] = not delete_raw


def compile_source_config_for_run(
    source_cfg: dict[str, Any],
    source_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply researcher-facing source selections to a source catalog/default config.
    """
    cfg = deepcopy(source_cfg)
    source_entry = source_entry or {}
    select_cfg = source_entry.get("select", {}) or {}

    if select_cfg:
        _compile_variable_selection(cfg, select_cfg)
        _compile_layer_selection(cfg, select_cfg)
        _compile_dimensions(cfg, select_cfg)
        _compile_aggregations(cfg, select_cfg)

    _compile_resampling_overrides(cfg, source_entry)
    _compile_processing_overrides(cfg, source_entry)
    _compile_download_overrides(cfg, source_entry)

    return cfg


def _expand_thermal_range_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = group.get("foreach", []) or []
    if not isinstance(rows, list):
        raise ConfigValidationError("thermal_range.foreach must be a list.")

    features: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            raise ConfigValidationError("Each thermal_range foreach item must be a dict.")

        source_id = row["source_id"]
        aggregation = row["aggregation"]

        name_parts = ["thermal_range", source_id, aggregation]
        for key in ["gcm", "ssp", "period"]:
            if row.get(key):
                name_parts.append(_sanitize_token(row[key]))

        base_query = {
            key: row[key]
            for key in ["gcm", "ssp", "period", "months"]
            if row.get(key) is not None
        }

        features.append(
            {
                "name": "_".join(_sanitize_token(item) for item in name_parts),
                "description": (
                    "Thermal range computed as tmax minus tmin from "
                    f"{source_id} {aggregation}."
                ),
                "expression": "tmax - tmin",
                "output_dtype": row.get("output_dtype", "float32"),
                "unit": row.get("unit", "degC"),
                "inputs": {
                    "tmax": {
                        "source_id": source_id,
                        "variable": row.get("tmax_variable", "tmax"),
                        "aggregation_name": aggregation,
                        **base_query,
                    },
                    "tmin": {
                        "source_id": source_id,
                        "variable": row.get("tmin_variable", "tmin"),
                        "aggregation_name": aggregation,
                        **base_query,
                    },
                },
            }
        )

    return features


def expand_derived_feature_groups(
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = deepcopy(run_cfg)
    groups = cfg.get("derived_feature_groups", []) or []
    if not groups:
        return cfg

    derived_features = list(cfg.get("derived_features", []) or [])

    for group in groups:
        recipe = group.get("recipe")
        if recipe == "thermal_range":
            derived_features.extend(_expand_thermal_range_group(group))
        else:
            raise ConfigValidationError(f"Unsupported derived feature recipe: {recipe}")

    cfg["derived_features"] = derived_features
    cfg.pop("derived_feature_groups", None)
    return cfg


def compile_run_config(
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Compile run-level convenience blocks that are independent of source loading.
    """
    return expand_derived_feature_groups(run_cfg)


def validate_researcher_run_config(
    run_cfg: dict[str, Any],
    run_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate old and simplified run configs without touching raster data.
    """
    errors: list[str] = []
    warnings: list[str] = []
    source_summaries: list[dict[str, Any]] = []

    try:
        compiled_run = compile_run_config(run_cfg)
        validate_run_config(compiled_run, run_config_path=run_config_path)
    except Exception as exc:
        return {
            "ok": False,
            "errors": [str(exc)],
            "warnings": [],
            "sources": [],
            "estimated_source_layers": 0,
            "estimated_derived_layers": 0,
            "estimated_layers": 0,
        }

    base_path = Path(run_config_path) if run_config_path is not None else None

    for source_entry in compiled_run.get("sources", []):
        try:
            source_config_path = resolve_path(
                source_entry["config"],
                base_path=base_path,
                must_exist=True,
            )
            source_cfg = load_yaml(source_config_path)
            source_cfg = expand_source_config(source_cfg)
            compiled_source = compile_source_config_for_run(source_cfg, source_entry)
            source_summary = _source_summary(source_entry, compiled_source)
            source_summary["config"] = str(source_config_path)
            source_summaries.append(source_summary)
        except Exception as exc:
            source_id = source_entry.get("id") or source_entry.get("config")
            errors.append(f"{source_id}: {exc}")

    estimated_source_layers = sum(
        item.get("estimated_layers", 0)
        for item in source_summaries
    )
    estimated_derived_layers = len(compiled_run.get("derived_features", []) or [])
    estimated_layers = estimated_source_layers + estimated_derived_layers

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "sources": source_summaries,
        "estimated_source_layers": estimated_source_layers,
        "estimated_derived_layers": estimated_derived_layers,
        "estimated_layers": estimated_layers,
    }


def _enabled_keys(collection: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in collection.items()
        if isinstance(value, dict) and bool(value.get("enabled", False))
    ]


def _source_summary(
    source_entry: dict[str, Any],
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    source = source_cfg.get("source", {}) or {}
    variables = _enabled_keys(source_cfg.get("variables", {}) or {})
    indices = _enabled_keys(source_cfg.get("indices", {}) or {})

    layer_count = len(variables) + len(indices)

    dimensions = {
        key: source_cfg.get(key, [])
        for key in ["gcms", "ssps", "periods"]
        if source_cfg.get(key)
    }
    dataset = source_cfg.get("dataset", {}) or {}
    layer_structure = dataset.get("layer_structure")
    output_dimension_keys = (
        ["gcms", "ssps", "periods"]
        if layer_structure == "future_monthly_multiband"
        else []
    )

    for key in output_dimension_keys:
        values = dimensions.get(key, [])
        layer_count *= max(1, len(values))

    aggregations = source_cfg.get("temporal_aggregations", []) or []
    if aggregations:
        layer_count = 0
        enabled_names = set(variables) | set(indices)
        for aggregation in aggregations:
            aggregation_variables = aggregation.get("variables") or sorted(enabled_names)
            applicable = set(aggregation_variables) & enabled_names
            n = max(0, len(applicable))
            for key in output_dimension_keys:
                values = dimensions.get(key, [])
                n *= max(1, len(values))
            layer_count += n

    vector_layers = 0
    for dataset_cfg in (source_cfg.get("datasets", {}) or {}).values():
        if not dataset_cfg.get("enabled", True):
            continue
        vector_layers += len(_enabled_keys(dataset_cfg.get("layers", {}) or {}))
    layer_count += vector_layers

    return {
        "id": source_entry.get("id") or source.get("id"),
        "provider": source.get("provider"),
        "product": source.get("product"),
        "variables": variables,
        "indices": indices,
        "dimensions": dimensions,
        "aggregations": [item.get("name") for item in aggregations],
        "vector_layers": vector_layers,
        "estimated_layers": layer_count,
    }


def load_and_compile_run_config(
    run_config_path: str | Path,
) -> dict[str, Any]:
    path = resolve_path(run_config_path, must_exist=True)
    cfg = load_yaml(path)
    return compile_run_config(cfg)


def render_run_config_yaml(
    run_cfg: dict[str, Any],
    compile_groups: bool = True,
) -> str:
    cfg = compile_run_config(run_cfg) if compile_groups else deepcopy(run_cfg)
    return yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
