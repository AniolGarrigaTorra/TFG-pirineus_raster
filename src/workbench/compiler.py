from __future__ import annotations

import json
from calendar import monthrange
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.io.config import load_yaml, resolve_path
from src.io.paths import get_grid_path
from src.pipeline.config import normalize_stages, validate_run_config
from src.pipeline.project_overrides import apply_run_overrides_to_project_cfg
from src.pipeline.source_overrides import (
    apply_run_overrides_to_source_cfg,
    normalize_source_domains,
)
from src.pipeline.variable_expansion import expand_source_config
from src.workbench.catalog import SUPPORTED_METRICS, SUPPORTED_RESAMPLING
from src.workbench.temporal import (
    MONTH_NAMES,
    SEASON_NAMES,
    infer_temporal_capability,
)


class ConfigValidationError(ValueError):
    """Raised when a researcher-facing run config is not valid."""


SUPPORTED_POSTPROCESS_METRICS = [
    "mean",
    "std",
    "min",
    "max",
    "count_threshold",
    "valid_observation_count",
]


def _format_source_validation_error(
    source_id: str,
    used_by: list[Any],
    error: Exception,
) -> str:
    lines = [f"Source '{source_id}' failed validation."]
    if used_by:
        lines.append(f"Used by final feature(s): {', '.join(map(str, used_by))}.")
    lines.append(f"Problem: {error}")
    return "\n".join(lines)


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


def _normalise_variable_group_reference(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("variable_groups."):
        return text.split(".", 1)[1]
    return text


def _is_yearly_static_collection(cfg: dict[str, Any]) -> bool:
    return cfg.get("dataset", {}).get("layer_structure") == "yearly_static_collection"


def _yearly_group_names(cfg: dict[str, Any]) -> set[str]:
    groups = set((cfg.get("variable_groups", {}) or {}).keys())
    for variable_cfg in (cfg.get("variables", {}) or {}).values():
        if isinstance(variable_cfg, dict) and variable_cfg.get("generated_from_group"):
            groups.add(str(variable_cfg["generated_from_group"]))
    return groups


def _expanded_variables_for_yearly_groups(
    cfg: dict[str, Any],
    selected: list[str],
) -> list[str]:
    variables = cfg.get("variables", {}) or {}
    group_names = _yearly_group_names(cfg)
    resolved: list[str] = []
    unknown: list[str] = []

    for raw_name in selected:
        name = _normalise_variable_group_reference(raw_name)
        if name in variables:
            resolved.append(name)
            continue
        if name in group_names:
            resolved.extend(
                variable_name
                for variable_name, variable_cfg in variables.items()
                if isinstance(variable_cfg, dict)
                and str(variable_cfg.get("generated_from_group")) == name
            )
            continue
        unknown.append(name)

    if unknown:
        available = sorted(set(variables) | group_names)
        raise ConfigValidationError(
            f"Unknown variables: {sorted(unknown)}. Available: {available}"
        )

    return sorted(set(resolved))


def _compile_variable_selection(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    selected_variables = [str(item) for item in _as_list(select_cfg.get("variables"))]
    selected_indices = [str(item) for item in _as_list(select_cfg.get("indices"))]

    variables = cfg.get("variables", {}) or {}
    indices = cfg.get("indices", {}) or {}

    if "variables" in select_cfg:
        if variables:
            if not selected_variables:
                _disable_all(variables)
            elif _is_yearly_static_collection(cfg):
                _enable_selected(
                    variables,
                    _expanded_variables_for_yearly_groups(cfg, selected_variables),
                    "variables",
                )
            else:
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
    configured_dimensions = cfg.get("dimensions", {}) or {}
    context_key_by_dimension = cfg.get("dimension_context_keys", {}) or {}

    for key, selected_values in dimensions_cfg.items():
        selected = [str(item) for item in _as_list(selected_values)]

        available = configured_dimensions.get(key, cfg.get(key))
        if available is None:
            raise ConfigValidationError(
                f"Source does not expose dimension {key!r}."
            )

        if not selected:
            cfg[key] = []
            continue

        available_values = [str(item) for item in available]
        unknown = sorted(set(selected) - set(available_values))
        if unknown:
            raise ConfigValidationError(
                f"Unknown values for dimension {key!r}: {unknown}. "
                f"Available: {available_values}"
            )

        if key in configured_dimensions:
            cfg.setdefault("dimensions", {})[key] = selected
        else:
            cfg[key] = selected

        context_key = context_key_by_dimension.get(key)
        if context_key:
            variables = cfg.get("variables", {}) or {}
            selected_set = set(selected)
            for variable_cfg in variables.values():
                if not isinstance(variable_cfg, dict):
                    continue
                context = variable_cfg.get("generation_context", {}) or {}
                if str(context.get(context_key)) not in selected_set:
                    variable_cfg["enabled"] = False


def _configured_category_values(variable_cfg: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in variable_cfg.get("category_classes", []) or []:
        if not isinstance(item, dict):
            continue
        raw_values = item.get("values")
        if raw_values is None and "value" in item:
            raw_values = [item["value"]]
        for value in _as_list(raw_values):
            values.add(str(value))
    return values


def _category_fraction_values(item: dict[str, Any]) -> list[Any]:
    values = item.get("class_values")
    if values is None and "class_value" in item:
        values = [item["class_value"]]
    values = _as_list(values)
    if not values:
        raise ConfigValidationError(f"Category fraction is missing class_values: {item}")
    return values


def _category_fraction_targets(
    cfg: dict[str, Any],
    variable: str,
) -> list[tuple[str, dict[str, Any]]]:
    variable = _normalise_variable_group_reference(variable)
    variables = cfg.get("variables", {}) or {}
    if variable in variables:
        return [(variable, variables[variable])]

    if not _is_yearly_static_collection(cfg) or variable not in _yearly_group_names(cfg):
        raise ConfigValidationError(
            f"Category fraction references unknown variable {variable!r}. "
            f"Available: {sorted(set(variables) | _yearly_group_names(cfg))}"
        )

    temporal_cfg = cfg.get("temporal", {}) or {}
    if temporal_cfg.get("output_mode") == "aggregate":
        raise ConfigValidationError(
            "Category fractions for yearly static collections require "
            "temporal output_mode='supplied_layers'. Build-time aggregation of "
            "categorical class fractions is not supported directly."
        )

    selected_years = set(
        int(year)
        for year in (temporal_cfg.get("layers", {}) or {}).get("years", []) or []
    )
    all_targets = [
        (name, variable_cfg)
        for name, variable_cfg in variables.items()
        if isinstance(variable_cfg, dict)
        and str(variable_cfg.get("generated_from_group")) == variable
    ]

    if selected_years:
        targets = [
            (name, variable_cfg)
            for name, variable_cfg in all_targets
            if _variable_reference_year(variable_cfg) in selected_years
        ]
    else:
        enabled_targets = [
            (name, variable_cfg)
            for name, variable_cfg in all_targets
            if bool(variable_cfg.get("enabled", False))
        ]
        targets = enabled_targets or all_targets

    if not targets:
        raise ConfigValidationError(
            f"Category fraction references yearly group {variable!r}, but no "
            "yearly variables match the temporal selection."
        )

    return sorted(
        targets,
        key=lambda item: _variable_reference_year(item[1]) or 0,
    )


def _category_fraction_name(
    *,
    item: dict[str, Any],
    variable: str,
    target_variable: str,
    target_cfg: dict[str, Any],
    class_values: list[Any],
    target_count: int,
) -> str:
    raw_name = item.get("name")
    year = _variable_reference_year(target_cfg)

    if raw_name:
        name = str(raw_name)
        if "{year}" in name and year is not None:
            return name.format(year=year)
        if target_count > 1 and year is not None:
            return f"{name}_{year}"
        return name

    values_token = _sanitize_token("_".join(map(str, class_values)))
    if target_count > 1 and year is not None:
        return f"{variable}_fraction_{values_token}_{year}"
    return f"{target_variable}_fraction_{values_token}"


def _compile_category_fractions(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    selected = select_cfg.get("category_fractions") or []
    if not selected:
        cfg.pop("category_fractions", None)
        return

    if not isinstance(selected, list):
        raise ConfigValidationError("select.category_fractions must be a list.")

    variables = cfg.get("variables", {}) or {}
    requested_output_variables = {
        str(item)
        for item in _as_list(select_cfg.get("variables"))
    }
    compiled: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for item in selected:
        if not isinstance(item, dict):
            raise ConfigValidationError("Each category fraction must be a dictionary.")

        variable = _normalise_variable_group_reference(item.get("variable", ""))
        targets = _category_fraction_targets(cfg, variable)
        class_values = _category_fraction_values(item)

        for target_variable, variable_cfg in targets:
            semantics = str(
                variable_cfg.get("value_semantics")
                or variable_cfg.get("data_type")
                or ""
            )
            if semantics not in {"categorical", "binary", "ordinal"}:
                raise ConfigValidationError(
                    f"Category fractions require a categorical/binary variable. "
                    f"{target_variable!r} has semantics {semantics!r}."
                )

            configured_values = _configured_category_values(variable_cfg)
            if configured_values:
                unknown = sorted(set(map(str, class_values)) - configured_values)
                if unknown:
                    raise ConfigValidationError(
                        f"Category fraction {item.get('name', variable)!r} uses "
                        f"unknown class values for {target_variable}: {unknown}. "
                        f"Available: {sorted(configured_values)}"
                    )

            name = _category_fraction_name(
                item=item,
                variable=variable,
                target_variable=target_variable,
                target_cfg=variable_cfg,
                class_values=class_values,
                target_count=len(targets),
            )
            if name in seen_names:
                raise ConfigValidationError(f"Duplicate category fraction name: {name}")
            seen_names.add(name)

            output_requested = (
                target_variable in requested_output_variables
                or variable in requested_output_variables
                or str(variable_cfg.get("generated_from_group", "")) in requested_output_variables
            )
            variable_cfg["enabled"] = True
            variable_cfg["build_output_enabled"] = bool(output_requested)

            compiled.append(
                {
                    "name": name,
                    "variable": target_variable,
                    "source_variable_group": (
                        variable
                        if variable != target_variable
                        else variable_cfg.get("generated_from_group")
                    ),
                    "class_values": class_values,
                    "label": item.get("label") or name,
                    "unit": "fraction",
                    "valid_range": [0, 1],
                    "data_type": "percentage",
                    "value_semantics": "fraction",
                    "resampling": item.get("resampling", "average"),
                }
            )

    cfg["category_fractions"] = compiled


def _aggregation_variable_names(cfg: dict[str, Any]) -> set[str]:
    names = set(cfg.get("variables", {}) or {}) | set(cfg.get("indices", {}) or {})
    if _is_yearly_static_collection(cfg):
        names |= _yearly_group_names(cfg)
    return names


def _variable_semantics(cfg: dict[str, Any], variable: str) -> set[str]:
    variable = _normalise_variable_group_reference(variable)
    variables = cfg.get("variables", {}) or {}
    targets: list[dict[str, Any]] = []

    if variable in variables and isinstance(variables[variable], dict):
        targets.append(variables[variable])
    elif _is_yearly_static_collection(cfg) and variable in _yearly_group_names(cfg):
        targets.extend(
            variable_cfg
            for variable_cfg in variables.values()
            if isinstance(variable_cfg, dict)
            and str(variable_cfg.get("generated_from_group")) == variable
        )

    semantics = {
        semantic
        for item in targets
        if (semantic := _normalise_value_semantics(item.get("value_semantics") or item.get("data_type")))
    }
    if not semantics and cfg.get("dataset", {}).get("data_type"):
        semantic = _normalise_value_semantics(cfg["dataset"]["data_type"])
        if semantic:
            semantics.add(semantic)
    return semantics


def _categorical_aggregation_variables(
    cfg: dict[str, Any],
    selected_variables: list[str],
) -> set[str]:
    categorical = {"categorical", "ordinal"}
    result: set[str] = set()
    for variable in selected_variables:
        semantics = _variable_semantics(cfg, variable)
        if semantics and semantics.issubset(categorical):
            result.add(variable)
    return result


def _validate_range_pair(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> list[int]:
    values = [int(item) for item in _as_list(value)]
    if len(values) != 2:
        raise ConfigValidationError(f"{name} must have exactly two values.")

    start, end = values
    if start > end:
        raise ConfigValidationError(f"{name} must be ordered start <= end.")

    if minimum is not None and start < minimum:
        raise ConfigValidationError(f"{name} starts before {minimum}: {values}")

    if maximum is not None and end > maximum:
        raise ConfigValidationError(f"{name} ends after {maximum}: {values}")

    return [start, end]


def _available_yearly_static_years(cfg: dict[str, Any]) -> list[int]:
    configured = cfg.get("years")
    if isinstance(configured, list) and configured:
        return sorted({int(item) for item in configured})

    years: set[int] = set()
    for variable_cfg in (cfg.get("variables", {}) or {}).values():
        if not isinstance(variable_cfg, dict):
            continue
        temporal = variable_cfg.get("temporal", {}) or {}
        if isinstance(temporal, dict) and temporal.get("reference_year") is not None:
            years.add(int(temporal["reference_year"]))
    return sorted(years)


def _validate_yearly_static_year_range(
    value: Any,
    *,
    name: str,
    cfg: dict[str, Any],
) -> list[int]:
    years = _validate_range_pair(value, name=name)
    available = _available_yearly_static_years(cfg)
    if not available:
        return years

    available_set = set(available)
    missing = [year for year in years if year not in available_set]
    if missing:
        raise ConfigValidationError(
            f"{name} endpoints must be available source years. "
            f"Got {years}; available years: {available}"
        )
    return years


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

    form = aggregation.get("form")
    layer_structure = cfg.get("dataset", {}).get("layer_structure")

    if layer_structure in {"monthly_climatology", "future_monthly_multiband"}:
        if form not in [None, "month_range_metric"]:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} uses form {form!r}, "
                f"but {layer_structure} only supports month_range_metric."
            )
        if "months" not in aggregation:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} must define months."
            )
        _validate_range_pair(
            aggregation["months"],
            name=f"Aggregation {aggregation['name']!r} months",
            minimum=1,
            maximum=12,
        )

    elif layer_structure == "monthly_time_series":
        if form not in [
            None,
            "year_range_month_range_metric",
            "year_then_across_years",
        ]:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} uses form {form!r}, "
                "but monthly time series only supports year range forms."
            )
        if "years" not in aggregation:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} must define years."
            )
        if "months" not in aggregation:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} must define months."
            )
        _validate_range_pair(
            aggregation["years"],
            name=f"Aggregation {aggregation['name']!r} years",
        )
        _validate_range_pair(
            aggregation["months"],
            name=f"Aggregation {aggregation['name']!r} months",
            minimum=1,
            maximum=12,
        )

    elif layer_structure == "yearly_static_collection":
        if form not in [None, "year_range_metric"]:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} uses form {form!r}, "
                "but yearly static collections only support year_range_metric."
            )
        if "years" not in aggregation:
            raise ConfigValidationError(
                f"Aggregation {aggregation['name']!r} must define years."
            )
        _validate_yearly_static_year_range(
            aggregation["years"],
            name=f"Aggregation {aggregation['name']!r} years",
            cfg=cfg,
        )

    elif layer_structure in {
        "static_single",
        "static_multi",
        "static_index_set",
        "vector_categorical",
        "osm_vector",
        "pdca_nested_zip_geotiff_collection",
        "temporal_aggregation",
    }:
        raise ConfigValidationError(
            f"Source layer_structure={layer_structure!r} does not support "
            "build-time temporal aggregations."
        )

    known_variables = _aggregation_variable_names(cfg)
    selected_variables = [str(item) for item in aggregation.get("variables", [])]
    if not selected_variables and _is_yearly_static_collection(cfg):
        selected_variables = sorted(_yearly_group_names(cfg))
    unknown = sorted(set(selected_variables) - known_variables)
    if unknown:
        raise ConfigValidationError(
            f"Aggregation {aggregation['name']!r} references unknown variables: {unknown}"
        )

    disallowed = _categorical_aggregation_variables(cfg, selected_variables)
    if disallowed:
        raise ConfigValidationError(
            f"Aggregation {aggregation['name']!r} cannot be applied directly to "
            f"categorical/ordinal variables: {sorted(disallowed)}. Select supplied "
            "layers or derive numeric category fractions first."
        )


def _compile_aggregation_selection(
    cfg: dict[str, Any],
    aggregation_select: Any,
) -> None:
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


def _postprocess_presets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    temporal_cfg = cfg.get("temporal_postprocess", {}) or {}
    presets = temporal_cfg.get("aggregation_presets", []) or []
    result: dict[str, dict[str, Any]] = {}

    if isinstance(presets, dict):
        for name, item in presets.items():
            if isinstance(item, dict):
                result[str(name)] = {"name": str(name), **deepcopy(item)}
        return result

    if isinstance(presets, list):
        for item in presets:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            result[str(item["name"])] = deepcopy(item)
        return result

    raise ConfigValidationError("temporal_postprocess.aggregation_presets must be a list or dictionary.")


def _month_list_from_selection(value: Any) -> list[int]:
    months = [int(item) for item in _as_list(value)]
    if not months:
        raise ConfigValidationError("Postprocess aggregation months cannot be empty.")

    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        raise ConfigValidationError(f"Invalid postprocess aggregation months: {invalid}")

    if len(months) == 2:
        start, end = months
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, 13)) + list(range(1, end + 1))

    seen: set[int] = set()
    result: list[int] = []
    for month in months:
        if month not in seen:
            result.append(month)
            seen.add(month)
    return result


def _postprocess_years(
    aggregation: dict[str, Any],
    temporal_cfg: dict[str, Any],
) -> list[int] | None:
    raw_years = aggregation.get("years")
    if raw_years is None:
        raw_years = temporal_cfg.get("default_years") or temporal_cfg.get("available_years")
    if raw_years is None:
        return None

    start, end = _validate_range_pair(
        raw_years,
        name=f"Postprocess aggregation {aggregation['name']!r} years",
    )
    available = temporal_cfg.get("available_years")
    if available is not None:
        available_start, available_end = _validate_range_pair(
            available,
            name="temporal_postprocess.available_years",
        )
        if start < available_start or end > available_end:
            raise ConfigValidationError(
                f"Postprocess aggregation {aggregation['name']!r} years "
                f"{[start, end]} are outside available source years "
                f"{[available_start, available_end]}."
            )
    return list(range(start, end + 1))


def _parse_iso_date(value: Any, *, name: str) -> date | None:
    if value in [None, ""]:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigValidationError(f"{name} must be YYYY-MM-DD: {value!r}") from exc


def _months_between_dates(start: date, end: date) -> list[int]:
    months: list[int] = []
    cursor_year = start.year
    cursor_month = start.month

    while (cursor_year, cursor_month) <= (end.year, end.month):
        months.append(cursor_month)
        cursor_month += 1
        if cursor_month > 12:
            cursor_month = 1
            cursor_year += 1

    seen: set[int] = set()
    result: list[int] = []
    for month in months:
        if month not in seen:
            result.append(month)
            seen.add(month)
    return result


def _postprocess_metric_defaults(method: str) -> dict[str, Any]:
    if method in {"mean", "std", "min", "max"}:
        return {
            "unit": "percent",
            "valid_range": [0, 100],
            "data_type": "percentage",
            "value_semantics": "percentage",
        }

    if method == "count_threshold":
        return {
            "unit": "days",
            "valid_range": [0, 366],
            "data_type": "continuous",
            "value_semantics": "count",
        }

    if method == "valid_observation_count":
        return {
            "unit": "observations",
            "valid_range": [0, 366],
            "data_type": "continuous",
            "value_semantics": "count",
        }

    return {}


def _postprocess_source_variable(
    aggregation: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    variables = cfg.get("variables", {}) or {}
    available = sorted(str(name) for name in variables)
    selected = [
        str(item)
        for item in _as_list(
            aggregation.get("source_variable")
            or aggregation.get("variable")
            or aggregation.get("variables")
        )
        if str(item).strip()
    ]

    if not selected:
        selected = [
            str(name)
            for name, variable_cfg in variables.items()
            if isinstance(variable_cfg, dict) and bool(variable_cfg.get("enabled", False))
        ]

    if not selected and len(available) == 1:
        selected = available

    if len(selected) != 1:
        raise ConfigValidationError(
            "Postprocess aggregations require exactly one source variable. "
            f"Got {selected or 'none'}; available: {available}"
        )

    source_variable = selected[0]
    if source_variable not in variables:
        raise ConfigValidationError(
            f"Unknown postprocess source variable {source_variable!r}. "
            f"Available: {available}"
        )

    return source_variable


def _normalise_postprocess_aggregation(
    aggregation: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if "name" not in aggregation:
        raise ConfigValidationError(f"Postprocess aggregation is missing name: {aggregation}")

    temporal_cfg = cfg.get("temporal_postprocess", {}) or {}
    supported = temporal_cfg.get("supported_methods") or SUPPORTED_POSTPROCESS_METRICS

    name = str(aggregation["name"]).strip()
    if not name:
        raise ConfigValidationError("Postprocess aggregation name cannot be empty.")

    method = str(aggregation.get("method") or aggregation.get("metric") or "mean")
    if method not in supported:
        raise ConfigValidationError(
            f"Unsupported postprocess metric {method!r}. Supported: {supported}"
        )

    exact_start = _parse_iso_date(
        aggregation.get("start_date"),
        name=f"Postprocess aggregation {name!r} start_date",
    )
    exact_end = _parse_iso_date(
        aggregation.get("end_date"),
        name=f"Postprocess aggregation {name!r} end_date",
    )
    if exact_start and exact_end and exact_start > exact_end:
        raise ConfigValidationError(
            f"Postprocess aggregation {name!r} start_date must be <= end_date."
        )

    if exact_start and exact_end:
        available = temporal_cfg.get("available_years")
        if available is not None:
            available_start, available_end = _validate_range_pair(
                available,
                name="temporal_postprocess.available_years",
            )
            min_date = date(available_start, 1, 1)
            max_date = date(available_end, 12, 31)
            if exact_start < min_date or exact_end > max_date:
                raise ConfigValidationError(
                    f"Postprocess aggregation {name!r} exact date range "
                    f"{exact_start.isoformat()} to {exact_end.isoformat()} "
                    f"is outside available source years "
                    f"{available_start}-{available_end}."
                )
        months = _months_between_dates(exact_start, exact_end)
        years = list(range(exact_start.year, exact_end.year + 1))
    else:
        months = _month_list_from_selection(
            aggregation.get("months") or temporal_cfg.get("default_months", [1, 12])
        )
        years = _postprocess_years({"name": name, **aggregation}, temporal_cfg)
    source_variable = _postprocess_source_variable(aggregation, cfg)
    base = {
        "filename": f"{name}.tif",
        "method": method,
        "source_variable": source_variable,
        "variables": [source_variable],
        "months": months,
        "description": aggregation.get("description") or f"{method} temporal postprocess aggregation.",
        "native_resolution_m": aggregation.get(
            "native_resolution_m",
            cfg.get("dataset", {}).get("native_resolution_m"),
        ),
        "temporal": {
            "type": "download_postprocess_aggregation",
            "months": months,
            "years": [min(years), max(years)] if years else None,
        },
        "required": bool(aggregation.get("required", False)),
    }
    base.update(_postprocess_metric_defaults(method))

    for key in [
        "unit",
        "valid_range",
        "data_type",
        "value_semantics",
        "scale_factor",
        "round_values",
    ]:
        if key in aggregation:
            base[key] = aggregation[key]

    if method == "count_threshold":
        base["threshold"] = float(aggregation.get("threshold", 50))
        base["comparison"] = str(aggregation.get("comparison", ">="))

    if years:
        base["years"] = [min(years), max(years)]
    if exact_start:
        base["start_date"] = exact_start.isoformat()
        base["temporal"]["start_date"] = exact_start.isoformat()
    if exact_end:
        base["end_date"] = exact_end.isoformat()
        base["temporal"]["end_date"] = exact_end.isoformat()

    return base


def _selected_postprocess_date_bounds(
    output_variables: dict[str, dict[str, Any]],
) -> tuple[date, date] | None:
    dates: list[date] = []

    for variable_cfg in output_variables.values():
        exact_start = _parse_iso_date(
            variable_cfg.get("start_date"),
            name=f"Postprocess output {variable_cfg.get('filename', '')} start_date",
        )
        exact_end = _parse_iso_date(
            variable_cfg.get("end_date"),
            name=f"Postprocess output {variable_cfg.get('filename', '')} end_date",
        )
        if exact_start and exact_end:
            dates.extend([exact_start, exact_end])
            continue

        years_value = variable_cfg.get("years")
        if not years_value:
            continue

        years = range(int(years_value[0]), int(years_value[1]) + 1)
        months = _month_list_from_selection(variable_cfg.get("months", [1, 12]))

        for year in years:
            for month in months:
                dates.append(date(year, month, 1))
                dates.append(date(year, month, monthrange(year, month)[1]))

    if not dates:
        return None

    return min(dates), max(dates)


def _iso_start(value: date) -> str:
    return f"{value.isoformat()}T00:00:00.000Z"


def _iso_end(value: date) -> str:
    return f"{value.isoformat()}T23:59:59.999Z"


def _apply_postprocess_download_bounds(
    cfg: dict[str, Any],
    output_variables: dict[str, dict[str, Any]],
) -> None:
    bounds = _selected_postprocess_date_bounds(output_variables)
    if bounds is None:
        return

    start, end = bounds
    temporal_cfg = cfg.setdefault("temporal_postprocess", {})
    temporal_cfg["start_date"] = start.isoformat()
    temporal_cfg["end_date"] = end.isoformat()

    day_count = max(1, (end - start).days + 1)
    per_day_estimate = int(temporal_cfg.get("hda_max_results_per_day_estimate", 0) or 0)
    estimated_max_results = day_count * per_day_estimate if per_day_estimate > 0 else None

    files_cfg = cfg.setdefault("download", {}).setdefault("files", {})
    for file_cfg in files_cfg.values():
        if not isinstance(file_cfg, dict):
            continue
        query = file_cfg.get("hda_query")
        if isinstance(query, dict):
            query["startdate"] = _iso_start(start)
            query["enddate"] = _iso_end(end)
        if estimated_max_results is not None:
            file_cfg["max_results"] = max(
                int(file_cfg.get("max_results", 0) or 0),
                estimated_max_results,
            )


def _compile_postprocess_aggregation_selection(
    cfg: dict[str, Any],
    temporal_select: dict[str, Any],
) -> None:
    temporal_cfg = cfg.setdefault("temporal_postprocess", {})
    aggregation_select = temporal_select.get("aggregations")

    if aggregation_select is None:
        existing = temporal_cfg.get("output_variables", {}) or {}
        if existing:
            return
        raise ConfigValidationError(
            "Temporal output_mode='postprocess_aggregate' requires at least one "
            "postprocess aggregation."
        )

    if isinstance(aggregation_select, list):
        use = [str(item) for item in aggregation_select]
        custom: list[dict[str, Any]] = []
    elif isinstance(aggregation_select, dict):
        use = [str(item) for item in _as_list(aggregation_select.get("use"))]
        custom = list(aggregation_select.get("custom", []) or [])
    else:
        raise ConfigValidationError("select.temporal.aggregations must be a list or dictionary.")

    presets = _postprocess_presets(cfg)
    unknown = sorted(set(use) - set(presets))
    if unknown:
        raise ConfigValidationError(
            f"Unknown postprocess aggregation presets: {unknown}. Available: {sorted(presets)}"
        )

    selected = [deepcopy(presets[name]) for name in use]
    selected.extend(deepcopy(custom))
    if not selected:
        raise ConfigValidationError(
            "Temporal output_mode='postprocess_aggregate' requires at least one "
            "selected preset or custom aggregation."
        )

    output_variables: dict[str, dict[str, Any]] = {}
    for aggregation in selected:
        if not isinstance(aggregation, dict):
            raise ConfigValidationError("Each postprocess aggregation must be a dictionary.")
        name = str(aggregation.get("name", "")).strip()
        if name in output_variables:
            raise ConfigValidationError(f"Duplicate postprocess aggregation name: {name}")
        output_variables[name] = _normalise_postprocess_aggregation(aggregation, cfg)

    temporal_cfg["output_variables"] = output_variables
    _apply_postprocess_download_bounds(cfg, output_variables)

    for variable_cfg in (cfg.get("variables", {}) or {}).values():
        if isinstance(variable_cfg, dict):
            variable_cfg["enabled"] = False


def _compile_aggregations(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    _compile_aggregation_selection(cfg, select_cfg.get("aggregations"))


def _selected_temporal_layers(layers_cfg: dict[str, Any]) -> dict[str, Any]:
    months = [str(item).lower() for item in _as_list(layers_cfg.get("months"))]
    seasons = [str(item).lower() for item in _as_list(layers_cfg.get("seasons"))]
    years = [int(item) for item in _as_list(layers_cfg.get("years"))]

    invalid_months = sorted(set(months) - set(MONTH_NAMES))
    if invalid_months:
        raise ConfigValidationError(
            f"Unknown temporal month layers: {invalid_months}. "
            f"Available: {MONTH_NAMES}"
        )

    invalid_seasons = sorted(set(seasons) - set(SEASON_NAMES))
    if invalid_seasons:
        raise ConfigValidationError(
            f"Unknown temporal season layers: {invalid_seasons}. "
            f"Available: {SEASON_NAMES}"
        )

    selected: dict[str, Any] = {}
    if "annual" in layers_cfg:
        selected["annual"] = bool(layers_cfg["annual"])
    if "annual_index" in layers_cfg:
        selected["annual_index"] = bool(layers_cfg["annual_index"])
    if "months" in layers_cfg:
        selected["months"] = months
    if "seasons" in layers_cfg:
        selected["seasons"] = seasons
    if "years" in layers_cfg:
        selected["years"] = years
    return selected


def _variable_reference_year(variable_cfg: dict[str, Any]) -> int | None:
    temporal = variable_cfg.get("temporal", {}) or {}
    if not isinstance(temporal, dict) or temporal.get("reference_year") is None:
        return None
    return int(temporal["reference_year"])


def _enable_static_year_variables(
    cfg: dict[str, Any],
    selected_years: list[int] | None,
    *,
    require_match: bool = True,
) -> None:
    if selected_years is None:
        return

    selected = set(selected_years)
    variables = cfg.get("variables", {}) or {}
    matched = False

    for variable_cfg in variables.values():
        if not isinstance(variable_cfg, dict):
            continue
        year = _variable_reference_year(variable_cfg)
        if year is None:
            continue
        variable_cfg["enabled"] = year in selected and bool(variable_cfg.get("enabled", True))
        matched = matched or bool(variable_cfg["enabled"])

    if not selected:
        return

    if require_match and not matched:
        raise ConfigValidationError(
            f"No enabled variables match selected temporal years: {sorted(selected)}"
        )


def _enable_yearly_aggregation_variables(cfg: dict[str, Any]) -> None:
    aggregations = cfg.get("temporal_aggregations", []) or []
    if not aggregations:
        return

    variables = cfg.get("variables", {}) or {}
    selected_names: set[str] = set()
    selected_years: set[int] = set()
    enabled_groups = {
        str(variable_cfg.get("generated_from_group"))
        for variable_cfg in variables.values()
        if isinstance(variable_cfg, dict)
        and bool(variable_cfg.get("enabled", False))
        and variable_cfg.get("generated_from_group")
    }
    default_groups = sorted(enabled_groups or _yearly_group_names(cfg))

    for aggregation in aggregations:
        aggregation_variables = [
            str(item)
            for item in _as_list(
                aggregation.get("variables") or default_groups
            )
        ]
        years = _validate_yearly_static_year_range(
            aggregation["years"],
            name=f"Aggregation {aggregation['name']!r} years",
            cfg=cfg,
        )
        selected_years.update(range(years[0], years[1] + 1))
        selected_names.update(_expanded_variables_for_yearly_groups(cfg, aggregation_variables))

    if not selected_names:
        raise ConfigValidationError("Yearly aggregation has no selected variables.")

    matched = False
    for variable, variable_cfg in variables.items():
        if not isinstance(variable_cfg, dict):
            continue
        year = _variable_reference_year(variable_cfg)
        enabled = (
            bool(variable_cfg.get("enabled", True))
            and variable in selected_names
            and year in selected_years
        )
        variable_cfg["enabled"] = enabled
        matched = matched or enabled

    if not matched:
        raise ConfigValidationError(
            "No yearly variables match the selected aggregation variables and years."
        )


def _compile_temporal_selection(
    cfg: dict[str, Any],
    select_cfg: dict[str, Any],
) -> None:
    temporal_select = select_cfg.get("temporal")
    if temporal_select is None:
        return

    if not isinstance(temporal_select, dict):
        raise ConfigValidationError("select.temporal must be a dictionary.")

    capability = infer_temporal_capability(cfg)
    output_mode = temporal_select.get(
        "output_mode",
        capability.get("default_output_mode", "static"),
    )
    output_modes = capability.get("output_modes", [])
    if output_mode not in output_modes:
        raise ConfigValidationError(
            f"Temporal output_mode {output_mode!r} is not supported for "
            f"{capability.get('kind')}. Available: {output_modes}"
        )

    temporal_cfg = cfg.setdefault("temporal", {})
    temporal_cfg["output_mode"] = output_mode

    if output_mode == "static":
        if temporal_select.get("aggregations"):
            raise ConfigValidationError("Static sources do not support aggregations.")
        return

    if output_mode == "aggregate":
        _compile_aggregation_selection(
            cfg,
            temporal_select.get("aggregations"),
        )
        if not cfg.get("temporal_aggregations"):
            raise ConfigValidationError(
                "Temporal output_mode='aggregate' requires at least one aggregation."
            )
        if capability.get("kind") == "yearly_static_collection":
            _enable_yearly_aggregation_variables(cfg)
        return

    if output_mode == "raw_slices":
        raw_cfg: dict[str, Any] = {}

        if "months" in temporal_select:
            raw_cfg["months"] = _validate_range_pair(
                temporal_select["months"],
                name="select.temporal.months",
                minimum=1,
                maximum=12,
            )
        else:
            raw_cfg["months"] = capability.get("default_months", [1, 12])

        if capability.get("kind") == "year_month_series":
            if "years" in temporal_select:
                raw_cfg["years"] = _validate_range_pair(
                    temporal_select["years"],
                    name="select.temporal.years",
                )
            else:
                raw_cfg["years"] = (
                    capability.get("default_years")
                    or capability.get("available_years")
                )
            if not raw_cfg.get("years"):
                raise ConfigValidationError(
                    "Raw year-month slices require select.temporal.years."
                )

        temporal_cfg["raw_slices"] = raw_cfg
        cfg["temporal_aggregations"] = []
        return

    if output_mode == "supplied_layers":
        layers = temporal_select.get("layers", {}) or {}
        if not isinstance(layers, dict):
            raise ConfigValidationError("select.temporal.layers must be a dictionary.")
        temporal_layers = _selected_temporal_layers(layers)
        temporal_cfg["layers"] = temporal_layers
        if capability.get("kind") == "yearly_static_collection":
            _enable_static_year_variables(
                cfg,
                [int(item) for item in temporal_layers["years"]]
                if "years" in temporal_layers
                else None,
                require_match=not bool(select_cfg.get("category_fractions")),
            )
        return

    if output_mode == "postprocess_aggregate":
        if temporal_select.get("raw_timesteps"):
            raise ConfigValidationError(
                "Raw timestep export is not implemented for this temporal product yet."
            )
        _compile_postprocess_aggregation_selection(cfg, temporal_select)
        return

    raise ConfigValidationError(f"Unsupported temporal output_mode: {output_mode}")


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
        fractions = cfg.get("category_fractions", []) or []
        fractions_by_name = {
            str(item.get("name")): item
            for item in fractions
            if isinstance(item, dict) and item.get("name")
        }
        for variable, method in by_variable.items():
            if method not in SUPPORTED_RESAMPLING:
                raise ConfigValidationError(
                    f"Unsupported resampling method for {variable}: {method}"
                )
            if str(variable) in fractions_by_name:
                fractions_by_name[str(variable)]["resampling"] = method
                continue
            variable_names = (
                _expanded_variables_for_yearly_groups(cfg, [str(variable)])
                if _is_yearly_static_collection(cfg)
                else [str(variable)]
            )
            for variable_name in variable_names:
                target[variable_name] = method


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


def _apply_source_resolution_metadata(cfg: dict[str, Any]) -> None:
    processing = cfg.get("processing", {}) or {}
    source_resolution = processing.get("source_resolution")
    if source_resolution is None:
        return

    source_resolution = str(source_resolution)
    source = cfg.setdefault("source", {})
    dataset = cfg.setdefault("dataset", {})

    crs_by_resolution = processing.get("source_crs_by_resolution", {}) or {}
    if source_resolution in crs_by_resolution:
        source["source_crs"] = str(crs_by_resolution[source_resolution])

    native_by_resolution = processing.get("native_resolution_m_by_resolution", {}) or {}
    if source_resolution in native_by_resolution:
        dataset["native_resolution_m"] = int(native_by_resolution[source_resolution])

    native_resolution_by_resolution = (
        processing.get("native_resolution_by_resolution", {}) or {}
    )
    if source_resolution in native_resolution_by_resolution:
        dataset["native_resolution"] = str(
            native_resolution_by_resolution[source_resolution]
        )


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
        _compile_temporal_selection(cfg, select_cfg)
        _compile_category_fractions(cfg, select_cfg)
        if "temporal" not in select_cfg:
            _compile_aggregations(cfg, select_cfg)

    _compile_resampling_overrides(cfg, source_entry)
    _compile_processing_overrides(cfg, source_entry)
    _apply_source_resolution_metadata(cfg)
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


def _derived_query_from_row(
    row: dict[str, Any],
    *,
    variable: str,
) -> dict[str, Any]:
    query = {
        "source_id": row["source_id"],
        "variable": row.get(f"{variable}_variable", variable),
    }
    for key in ["aggregation", "aggregation_name", "gcm", "ssp", "period", "months"]:
        if row.get(key) is not None:
            query["aggregation_name" if key == "aggregation" else key] = row[key]
    return query


def _expand_simple_recipe_group(
    group: dict[str, Any],
) -> list[dict[str, Any]]:
    recipe = str(group.get("recipe"))
    rows = group.get("foreach", []) or []
    if not isinstance(rows, list):
        raise ConfigValidationError(f"{recipe}.foreach must be a list.")

    specs = {
        "water_balance": {
            "inputs": ["prec", "pet"],
            "unit": "mm",
            "description": "Water balance computed as precipitation minus potential evapotranspiration.",
        },
        "aridity_index": {
            "inputs": ["prec", "pet"],
            "unit": "ratio",
            "description": "Aridity index computed from precipitation and potential evapotranspiration.",
        },
        "snow_persistence_ratio": {
            "inputs": ["snow_days", "valid_days"],
            "unit": "ratio",
            "description": "Snow persistence ratio computed as snow days divided by valid observation days.",
        },
        "seasonal_contrast": {
            "inputs": ["a", "b"],
            "unit": "source_units",
            "description": "Contrast between two selected environmental layers.",
        },
    }
    if recipe not in specs:
        raise ConfigValidationError(f"Unsupported recipe group: {recipe}")

    features: list[dict[str, Any]] = []
    spec = specs[recipe]
    for row in rows:
        if not isinstance(row, dict):
            raise ConfigValidationError(f"Each {recipe} foreach item must be a dict.")
        name = row.get("name") or "_".join(
            _sanitize_token(item)
            for item in [recipe, row["source_id"], row.get("aggregation", row.get("aggregation_name", "layer"))]
        )
        features.append(
            {
                "name": name,
                "operation": "recipe",
                "recipe": recipe,
                "description": row.get("description", spec["description"]),
                "unit": row.get("unit", spec["unit"]),
                "output_dtype": row.get("output_dtype", "float32"),
                "parameters": row.get("parameters", {}),
                "inputs": {
                    input_name: _derived_query_from_row(row, variable=input_name)
                    for input_name in spec["inputs"]
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
        elif recipe in {
            "water_balance",
            "aridity_index",
            "snow_persistence_ratio",
            "seasonal_contrast",
        }:
            derived_features.extend(_expand_simple_recipe_group(group))
        else:
            raise ConfigValidationError(f"Unsupported derived feature recipe: {recipe}")

    cfg["derived_features"] = derived_features
    cfg.pop("derived_feature_groups", None)
    return cfg


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _feature_build_type(feature: dict[str, Any]) -> str:
    value = str(feature.get("build_type") or feature.get("kind") or "").strip()
    if not value:
        raise ConfigValidationError(f"Feature is missing build_type: {feature}")
    return value


def _extract_feature_aggregation_names(feature: dict[str, Any]) -> list[str]:
    """Extract aggregation names from a feature's source temporal configuration."""
    source = feature.get("source")
    if not isinstance(source, dict):
        return []
    
    temporal = source.get("temporal")
    if not isinstance(temporal, dict):
        return []
    
    aggregations_cfg = temporal.get("aggregations")
    if not aggregations_cfg:
        return []
    
    names: list[str] = []
    
    # Extract "use" aggregations (preset names)
    if isinstance(aggregations_cfg, dict):
        use_list = aggregations_cfg.get("use") or []
        if not isinstance(use_list, list):
            use_list = [use_list] if use_list else []
        names.extend([str(item) for item in use_list])
        
        # Extract "custom" aggregations (custom definitions)
        custom_list = aggregations_cfg.get("custom") or []
        if not isinstance(custom_list, list):
            custom_list = [custom_list] if custom_list else []
        for custom_agg in custom_list:
            if isinstance(custom_agg, dict):
                agg_name = custom_agg.get("name")
                if agg_name:
                    names.append(str(agg_name))
    elif isinstance(aggregations_cfg, list):
        # Fallback: list of aggregation names or dicts
        for item in aggregations_cfg:
            if isinstance(item, dict):
                agg_name = item.get("name")
                if agg_name:
                    names.append(str(agg_name))
            else:
                names.append(str(item))
    
    return names


def _feature_outputs(feature: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = feature.get("outputs")
    if outputs is None:
        # Check if this feature should be expanded based on temporal aggregations
        aggregation_names = _extract_feature_aggregation_names(feature)
        
        if aggregation_names:
            # Create one output per aggregation
            base_output = {
                key: deepcopy(value)
                for key, value in feature.items()
                if key not in {"outputs"}
            }
            
            expanded_outputs = []
            source = base_output.get("source")
            if isinstance(source, dict):
                for agg_name in aggregation_names:
                    output = deepcopy(base_output)
                    output_source = output.get("source")
                    if isinstance(output_source, dict):
                        # Update query to use aggregation_name
                        query = output_source.get("query") or {}
                        if not isinstance(query, dict):
                            query = {}
                        query["aggregation_name"] = agg_name
                        output_source["query"] = query
                        
                        # Update output name to include aggregation suffix
                        # Use suffix so it gets applied by _feature_output_name
                        if "name" not in output:
                            output["suffix"] = agg_name
                        
                        # Update temporal config to only include this aggregation
                        temporal = output_source.get("temporal")
                        if isinstance(temporal, dict):
                            aggregations_cfg = temporal.get("aggregations")
                            if isinstance(aggregations_cfg, dict):
                                # Keep all aggregations but they'll be filtered at build time
                                # The aggregation_name in query tells the build which one to use
                                pass
                    expanded_outputs.append(output)
            
            return expanded_outputs if expanded_outputs else [base_output]
        else:
            # No aggregations, use the original logic
            output = {
                key: deepcopy(value)
                for key, value in feature.items()
                if key not in {"outputs"}
            }
            return [output]
    
    if not isinstance(outputs, list) or not outputs:
        raise ConfigValidationError(
            f"Feature {feature.get('name')!r} outputs must be a non-empty list."
        )
    return [deepcopy(output) for output in outputs]


def _feature_output_name(feature: dict[str, Any], output: dict[str, Any]) -> str:
    name = str(output.get("name") or feature.get("name") or "").strip()
    suffix = str(output.get("suffix") or "").strip()
    if suffix and name == str(feature.get("name") or "").strip():
        name = f"{name}_{_sanitize_token(suffix)}"
    if not name:
        raise ConfigValidationError(f"Feature output is missing name: {feature}")
    return _sanitize_token(name)


def _source_input_select(input_cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(input_cfg.get("select"), dict):
        return deepcopy(input_cfg["select"])

    select: dict[str, Any] = {}

    category_fraction = input_cfg.get("category_fraction")
    if isinstance(category_fraction, dict):
        select["variables"] = []
        select["category_fractions"] = [deepcopy(category_fraction)]
    elif input_cfg.get("layer") is not None:
        select["layers"] = [str(input_cfg["layer"])]
    elif input_cfg.get("variable") is not None:
        select["variables"] = [str(input_cfg["variable"])]

    if isinstance(input_cfg.get("dimensions"), dict):
        select["dimensions"] = deepcopy(input_cfg["dimensions"])

    if isinstance(input_cfg.get("temporal"), dict):
        select["temporal"] = deepcopy(input_cfg["temporal"])

    if not select:
        raise ConfigValidationError(f"Source input is missing select information: {input_cfg}")

    return select


def _source_input_overrides(input_cfg: dict[str, Any]) -> dict[str, Any] | None:
    overrides = deepcopy(input_cfg.get("overrides") or {})
    processing: dict[str, Any] = deepcopy(overrides.get("processing") or {})
    resampling: dict[str, Any] = deepcopy(overrides.get("resampling") or {})
    download: dict[str, Any] = deepcopy(overrides.get("download") or {})

    if input_cfg.get("source_resolution") is not None:
        processing["source_resolution"] = str(input_cfg["source_resolution"])
    if input_cfg.get("target_resolution_m") is not None:
        processing["target_resolution_m"] = int(input_cfg["target_resolution_m"])
    if input_cfg.get("resampling") is not None:
        category_fraction = input_cfg.get("category_fraction")
        variable = (
            category_fraction.get("name")
            if isinstance(category_fraction, dict) and category_fraction.get("name")
            else input_cfg.get("variable") or input_cfg.get("output_variable") or input_cfg.get("layer")
        )
        if variable is None and isinstance(input_cfg.get("query"), dict):
            variable = input_cfg.get("query", {}).get("variable")
        if variable:
            resampling.setdefault("by_variable", {})[str(variable)] = str(input_cfg["resampling"])
    if input_cfg.get("keep_raw_after_clip") is not None:
        download["keep_raw_after_clip"] = bool(input_cfg["keep_raw_after_clip"])

    compiled = {
        "processing": processing or None,
        "resampling": resampling or None,
        "download": download or None,
    }
    compact = {key: value for key, value in compiled.items() if value is not None}
    return compact or None


def _mergeable_temporal_key(temporal: Any) -> Any:
    if not isinstance(temporal, dict):
        return temporal
    output_mode = temporal.get("output_mode")
    key: dict[str, Any] = {"output_mode": output_mode}
    if output_mode == "raw_slices":
        for item in ["months", "years"]:
            if item in temporal:
                key[item] = deepcopy(temporal[item])
    return key


def _mergeable_select_key(select: dict[str, Any]) -> dict[str, Any]:
    key_select = deepcopy(select)
    for key in ["variables", "indices", "layers", "category_fractions", "dimensions"]:
        key_select.pop(key, None)
    if "temporal" in key_select:
        key_select["temporal"] = _mergeable_temporal_key(key_select["temporal"])
    return key_select


def _mergeable_overrides_key(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    if not overrides:
        return None

    key_overrides = deepcopy(overrides)
    resampling = key_overrides.get("resampling")
    if isinstance(resampling, dict):
        for key in ["by_variable", "variables", "per_variable"]:
            resampling.pop(key, None)
        if not resampling:
            key_overrides.pop("resampling", None)

    return key_overrides or None


def _merge_unique_list(existing: list[Any], incoming: list[Any]) -> list[Any]:
    result = list(existing)
    seen = {_canonical_json(item) for item in result}
    for item in incoming:
        key = _canonical_json(item)
        if key not in seen:
            result.append(deepcopy(item))
            seen.add(key)
    return result


def _merge_dict_unique(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(existing)
    for key, value in incoming.items():
        if key not in result:
            result[key] = deepcopy(value)
            continue
        if _canonical_json(result[key]) != _canonical_json(value):
            raise ConfigValidationError(
                f"Conflicting values for source requirement key {key!r}: "
                f"{result[key]!r} vs {value!r}"
            )
    return result


def _merge_dimension_select(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(existing)
    for key, value in incoming.items():
        result[key] = _merge_unique_list(_as_list(result.get(key)), _as_list(value))
    return result


def _merge_temporal_layers(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(existing)
    for key, value in incoming.items():
        if isinstance(value, bool):
            result[key] = bool(result.get(key, False)) or value
        elif isinstance(value, list):
            result[key] = _merge_unique_list(_as_list(result.get(key)), value)
        elif value is not None:
            result.setdefault(key, deepcopy(value))
    return result


def _merge_temporal_aggregations(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(existing)
    if "use" in incoming:
        result["use"] = _merge_unique_list(
            _as_list(result.get("use")),
            _as_list(incoming.get("use")),
        )

    if "custom" in incoming:
        by_name: dict[str, dict[str, Any]] = {}
        unnamed_existing: list[Any] = []
        for item in _as_list(result.get("custom")):
            if isinstance(item, dict) and item.get("name"):
                by_name[str(item["name"])] = deepcopy(item)
            else:
                unnamed_existing.append(deepcopy(item))

        for item in _as_list(incoming.get("custom")):
            if isinstance(item, dict) and item.get("name"):
                name = str(item["name"])
                previous = by_name.get(name)
                if (
                    previous is not None
                    and _canonical_json(previous) != _canonical_json(item)
                ):
                    raise ConfigValidationError(
                        f"Conflicting temporal aggregation named {name!r}."
                    )
                by_name[name] = deepcopy(item)
            else:
                unnamed_existing = _merge_unique_list(unnamed_existing, [item])

        result["custom"] = unnamed_existing + list(by_name.values())
    return result


def _merge_temporal_select(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(existing)
    existing_mode = result.get("output_mode")
    incoming_mode = incoming.get("output_mode")
    if existing_mode != incoming_mode:
        raise ConfigValidationError(
            f"Cannot merge temporal output modes {existing_mode!r} and {incoming_mode!r}."
        )

    mode = incoming_mode
    if mode == "supplied_layers":
        result["layers"] = _merge_temporal_layers(
            result.get("layers", {}) or {},
            incoming.get("layers", {}) or {},
        )
    elif mode in {"aggregate", "postprocess_aggregate"}:
        result["aggregations"] = _merge_temporal_aggregations(
            result.get("aggregations", {}) or {},
            incoming.get("aggregations", {}) or {},
        )
    else:
        for key, value in incoming.items():
            if key not in result:
                result[key] = deepcopy(value)
            elif _canonical_json(result[key]) != _canonical_json(value):
                raise ConfigValidationError(
                    f"Cannot merge temporal selection key {key!r}: "
                    f"{result[key]!r} vs {value!r}"
                )
    return result


def _merge_source_select(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key in {"variables", "indices", "layers", "category_fractions"}:
            existing[key] = _merge_unique_list(_as_list(existing.get(key)), _as_list(value))
            continue
        if key == "dimensions" and isinstance(value, dict):
            existing[key] = _merge_dimension_select(existing.get(key, {}) or {}, value)
            continue
        if key == "temporal" and isinstance(value, dict):
            existing[key] = _merge_temporal_select(existing.get(key, {}) or {}, value)
            continue
        existing.setdefault(key, deepcopy(value))


def _merge_resampling_overrides(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(existing)
    for key, value in incoming.items():
        if key in {"by_variable", "variables", "per_variable"} and isinstance(value, dict):
            target = result.setdefault("by_variable", {})
            for variable, method in value.items():
                if variable in target and target[variable] != method:
                    raise ConfigValidationError(
                        f"Conflicting resampling for {variable!r}: "
                        f"{target[variable]!r} vs {method!r}"
                    )
                target[variable] = method
            continue
        if key in result and _canonical_json(result[key]) != _canonical_json(value):
            raise ConfigValidationError(
                f"Conflicting resampling override {key!r}: {result[key]!r} vs {value!r}"
            )
        result[key] = deepcopy(value)
    return result


def _merge_source_overrides(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not existing:
        return deepcopy(incoming) if incoming else None
    if not incoming:
        return deepcopy(existing)

    result = deepcopy(existing)
    for key, value in incoming.items():
        if key == "resampling" and isinstance(value, dict):
            result[key] = _merge_resampling_overrides(result.get(key, {}) or {}, value)
        elif key in {"processing", "download"} and isinstance(value, dict):
            result[key] = _merge_dict_unique(result.get(key, {}) or {}, value)
        elif key not in result:
            result[key] = deepcopy(value)
        elif _canonical_json(result[key]) != _canonical_json(value):
            raise ConfigValidationError(
                f"Conflicting source override {key!r}: {result[key]!r} vs {value!r}"
            )
    return {key: value for key, value in result.items() if value not in [None, {}, []]} or None


def _try_merge_source_requirement(
    entry: dict[str, Any],
    select: dict[str, Any],
    overrides: dict[str, Any] | None,
    used_by_feature: str | None = None,
) -> bool:
    try:
        merged_select = deepcopy(entry["select"])
        _merge_source_select(merged_select, select)
        merged_overrides = _merge_source_overrides(entry.get("overrides"), overrides)
    except ConfigValidationError:
        return False

    entry["select"] = merged_select
    if merged_overrides:
        entry["overrides"] = merged_overrides
    else:
        entry.pop("overrides", None)
    if used_by_feature:
        used_by = list(entry.get("used_by_features", []) or [])
        if used_by_feature not in used_by:
            used_by.append(used_by_feature)
        entry["used_by_features"] = used_by
    return True


def _source_requirement_alias(
    *,
    input_cfg: dict[str, Any],
    source_entries: list[dict[str, Any]],
    source_key_to_alias: dict[str, str],
    source_alias_to_entry: dict[str, dict[str, Any]],
    alias_counts: dict[str, int],
    used_by_feature: str | None = None,
) -> str:
    config = input_cfg.get("config")
    source_id = str(input_cfg.get("source_id") or input_cfg.get("id") or "").strip()
    if not config:
        raise ConfigValidationError(
            f"Source input {source_id or input_cfg!r} is missing config path."
        )
    if not source_id:
        source_id = Path(str(config)).stem

    select = _source_input_select(input_cfg)
    overrides = _source_input_overrides(input_cfg)
    stages = input_cfg.get("stages")
    key_payload = {
        "config": str(config),
        "select": _mergeable_select_key(select),
        "overrides": _mergeable_overrides_key(overrides),
        "stages": stages,
    }
    key = _canonical_json(key_payload)
    if key in source_key_to_alias:
        alias = source_key_to_alias[key]
        if _try_merge_source_requirement(source_alias_to_entry[alias], select, overrides, used_by_feature):
            return alias

        conflict_payload = {
            **key_payload,
            "overrides": overrides,
        }
        key = _canonical_json(conflict_payload)
        if key in source_key_to_alias:
            alias = source_key_to_alias[key]
            if _try_merge_source_requirement(source_alias_to_entry[alias], select, overrides, used_by_feature):
                return alias

    base_alias = _sanitize_token(source_id)
    alias_counts[base_alias] = alias_counts.get(base_alias, 0) + 1
    alias = base_alias if alias_counts[base_alias] == 1 else f"{base_alias}_req{alias_counts[base_alias]}"

    entry = {
        "id": alias,
        "config": str(config),
        "select": select,
    }
    if stages:
        entry["stages"] = stages
    if overrides:
        entry["overrides"] = overrides
    if used_by_feature:
        entry["used_by_features"] = [used_by_feature]

    source_entries.append(entry)
    source_key_to_alias[key] = alias
    source_alias_to_entry[alias] = entry
    return alias


def _source_query_from_input(
    input_cfg: dict[str, Any],
    *,
    source_alias: str,
) -> dict[str, Any]:
    query = deepcopy(input_cfg.get("query") or {})
    if not isinstance(query, dict):
        raise ConfigValidationError(f"Source input query must be a dictionary: {input_cfg}")

    if not query.get("variable"):
        category_fraction = input_cfg.get("category_fraction")
        if isinstance(category_fraction, dict) and category_fraction.get("name"):
            query["variable"] = str(category_fraction["name"])
        elif input_cfg.get("output_variable") is not None:
            query["variable"] = str(input_cfg["output_variable"])
        elif input_cfg.get("variable") is not None:
            query["variable"] = str(input_cfg["variable"])
        elif input_cfg.get("layer") is not None:
            layer_name = str(input_cfg["layer"])
            query["variable"] = layer_name.split(".")[-1]
        else:
            raise ConfigValidationError(f"Source input is missing output query variable: {input_cfg}")

    query["source_id"] = source_alias
    return query


def _compile_feature_input(
    input_cfg: dict[str, Any],
    *,
    source_entries: list[dict[str, Any]],
    source_key_to_alias: dict[str, str],
    source_alias_to_entry: dict[str, dict[str, Any]],
    alias_counts: dict[str, int],
    known_feature_outputs: set[str],
    used_by_feature: str | None = None,
) -> dict[str, Any]:
    input_kind = str(input_cfg.get("kind") or input_cfg.get("type") or "source")

    if input_kind == "feature":
        variable = str(input_cfg.get("output") or input_cfg.get("feature") or "").strip()
        if not variable:
            raise ConfigValidationError(f"Feature input is missing feature/output: {input_cfg}")
        if variable not in known_feature_outputs:
            raise ConfigValidationError(
                f"Feature input references unknown or later output {variable!r}."
            )
        return {"source_id": "derived", "variable": variable}

    if input_kind != "source":
        raise ConfigValidationError(f"Unsupported feature input kind {input_kind!r}.")

    source_alias = _source_requirement_alias(
        input_cfg=input_cfg,
        source_entries=source_entries,
        source_key_to_alias=source_key_to_alias,
        source_alias_to_entry=source_alias_to_entry,
        alias_counts=alias_counts,
        used_by_feature=used_by_feature,
    )
    return _source_query_from_input(input_cfg, source_alias=source_alias)


def _normalise_value_semantics(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"auto", "none", "null", "unknown"}:
        return None
    aliases = {
        "continuous": "intensive",
        "numeric": "intensive",
        "float": "intensive",
        "float32": "intensive",
        "float64": "intensive",
        "integer": "count",
        "int": "count",
        "boolean": "binary",
        "bool": "binary",
        "class": "categorical",
        "classes": "categorical",
        "proportion": "fraction",
        "coverage_fraction": "fraction",
    }
    return aliases.get(text.lower(), text)


def _source_input_variable_name(input_cfg: dict[str, Any]) -> str | None:
    category_fraction = input_cfg.get("category_fraction")
    if isinstance(category_fraction, dict) and category_fraction.get("variable") is not None:
        return _normalise_variable_group_reference(category_fraction["variable"])

    if input_cfg.get("variable") is not None:
        return _normalise_variable_group_reference(input_cfg["variable"])

    query = input_cfg.get("query")
    if isinstance(query, dict) and query.get("variable") is not None:
        return _normalise_variable_group_reference(query["variable"])

    if input_cfg.get("layer") is not None:
        return str(input_cfg["layer"]).split(".")[-1]

    return None


def _source_input_value_semantics(input_cfg: dict[str, Any]) -> str | None:
    if isinstance(input_cfg.get("category_fraction"), dict):
        return "fraction"

    config = input_cfg.get("config")
    variable = _source_input_variable_name(input_cfg)
    if not config or not variable:
        return None

    try:
        source_cfg = expand_source_config(load_yaml(resolve_path(config, must_exist=True)))
    except Exception:
        return None

    semantics = _variable_semantics(source_cfg, variable)
    if len(semantics) == 1:
        return next(iter(semantics))
    if len(semantics) > 1 and semantics.issubset({"categorical", "ordinal", "binary"}):
        return "categorical"
    return None


def _feature_input_value_semantics(
    input_cfg: Any,
    known_feature_semantics: dict[str, str],
) -> str | None:
    if not isinstance(input_cfg, dict):
        return None

    input_kind = str(input_cfg.get("kind") or input_cfg.get("type") or "source")
    if input_kind == "feature":
        variable = str(input_cfg.get("output") or input_cfg.get("feature") or "").strip()
        return known_feature_semantics.get(variable)

    if input_kind == "source":
        return _source_input_value_semantics(input_cfg)

    return None


def _feature_raw_inputs(feature: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    inputs = deepcopy(output.get("inputs") or feature.get("inputs") or {})
    build_type = _feature_build_type(feature)
    if build_type == "source_layer" and not inputs:
        source_input = output.get("source") or feature.get("source")
        if isinstance(source_input, dict):
            inputs = {"x": source_input}
    return inputs if isinstance(inputs, dict) else {}


def _infer_expression_value_semantics(
    expression: str,
    input_semantics: dict[str, str | None],
) -> str | None:
    text = str(expression or "").strip()
    compact = text.replace(" ", "")
    if compact == "x":
        return input_semantics.get("x")
    if any(op in compact for op in ["/", "log(", "log10("]):
        return "ratio"
    if any(op in compact for op in [">=", "<=", "==", "!=", ">", "<"]):
        if compact.startswith("where("):
            return input_semantics.get("x") or "intensive"
        return "binary"
    if any(op in compact for op in ["+", "-", "*", "**"]):
        values = {value for value in input_semantics.values() if value}
        if values and values.issubset({"percentage"}):
            return "percentage"
        if values and values.issubset({"fraction", "binary"}):
            return "fraction"
        if values and values.issubset({"count"}):
            return "count"
        return "intensive"
    return input_semantics.get("x")


def _infer_feature_value_semantics(
    feature: dict[str, Any],
    output: dict[str, Any],
    known_feature_semantics: dict[str, str],
) -> str | None:
    explicit = _normalise_value_semantics(output.get("value_semantics"))
    if explicit:
        return explicit
    explicit = _normalise_value_semantics(feature.get("value_semantics"))
    if explicit:
        return explicit

    build_type = _feature_build_type(feature)
    raw_inputs = _feature_raw_inputs(feature, output)
    input_semantics = {
        str(alias): _feature_input_value_semantics(input_cfg, known_feature_semantics)
        for alias, input_cfg in raw_inputs.items()
    }

    if build_type == "source_layer":
        return input_semantics.get("x")

    if build_type in {"masking", "recipe"}:
        recipe = str(output.get("recipe") or feature.get("recipe") or "")
        if recipe in {"binary_threshold_mask", "class_mask"}:
            return "binary"
        if recipe == "reclassification":
            return input_semantics.get("x") or "categorical"
        if recipe == "thermal_range":
            return "intensive"
        if recipe == "water_balance":
            return "intensive_depth"
        if recipe == "aridity_index":
            return "ratio"
        if recipe == "snow_persistence_ratio":
            return "fraction"
        if recipe == "seasonal_contrast":
            parameters = output.get("parameters") or feature.get("parameters") or {}
            metric = str(parameters.get("metric", "difference")) if isinstance(parameters, dict) else "difference"
            return "ratio" if "ratio" in metric else (input_semantics.get("a") or "intensive")

    if build_type in {"terrain", "focal", "distance", "spatial"}:
        operation = str(output.get("operation") or feature.get("operation") or "")
        if build_type != "spatial":
            operation = build_type
        method = str(output.get("method") or feature.get("method") or "")
        if operation == "distance":
            return "intensive"
        if operation == "terrain":
            return "circular" if method == "aspect" else "intensive"
        if operation == "focal":
            source_semantics = input_semantics.get("x")
            if method == "majority":
                return source_semantics or "categorical"
            if method == "diversity":
                return "count"
            if method == "mean" and source_semantics in {"binary", "fraction"}:
                return "fraction"
            if method in {"min", "max"} and source_semantics:
                return source_semantics
            if method == "sum" and source_semantics in {"count", "extensive"}:
                return source_semantics
            return "intensive"

    if build_type == "expression":
        return _infer_expression_value_semantics(
            str(output.get("expression") or feature.get("expression") or ""),
            input_semantics,
        )

    return None


def _infer_feature_output_dtype(
    feature: dict[str, Any],
    output: dict[str, Any],
    value_semantics: str | None,
) -> str:
    explicit = output.get("output_dtype") or feature.get("output_dtype")
    if explicit:
        return str(explicit)

    build_type = _feature_build_type(feature)
    recipe = str(output.get("recipe") or feature.get("recipe") or "")
    if build_type == "masking" and recipe in {"binary_threshold_mask", "class_mask"}:
        return "uint8"
    if value_semantics in {"categorical", "ordinal", "binary"} and build_type == "source_layer":
        return "int32"
    return "float32"


def _normalise_feature_evaluation_stage(value: Any) -> str | None:
    if value in [None, ""]:
        return None
    text = str(value).strip().lower()
    aliases = {
        "after_resampling": "target_grid",
        "after_resample": "target_grid",
        "post_resample": "target_grid",
        "post_resampling": "target_grid",
        "target": "target_grid",
        "native": "native_then_resample",
        "before_resample": "native_then_resample",
        "before_resampling": "native_then_resample",
        "pre_resample": "native_then_resample",
        "source_grid": "native_then_resample",
    }
    normalised = aliases.get(text, text)
    if normalised not in {"target_grid", "native_then_resample"}:
        raise ConfigValidationError(
            f"Unsupported evaluation_stage {value!r}. "
            "Use 'target_grid' or 'native_then_resample'."
        )
    return normalised


def _normalise_feature_evaluation_resolution(value: Any) -> Any:
    if value in [None, "", "native", "auto"]:
        return "native"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(
            f"evaluation_resolution_m must be 'native' or a positive metre value: {value!r}"
        ) from exc
    if number <= 0:
        raise ConfigValidationError(
            f"evaluation_resolution_m must be positive: {value!r}"
        )
    return int(number) if number.is_integer() else number


def _normalise_feature_post_resampling(value: Any) -> str | None:
    if value in [None, ""]:
        return None
    text = str(value).strip()
    if text not in SUPPORTED_RESAMPLING:
        raise ConfigValidationError(
            f"Unsupported post_resampling method {text!r}. "
            f"Supported: {SUPPORTED_RESAMPLING}"
        )
    return text


def _normalise_feature_inputs(
    feature: dict[str, Any],
    output: dict[str, Any],
    *,
    source_entries: list[dict[str, Any]],
    source_key_to_alias: dict[str, str],
    source_alias_to_entry: dict[str, dict[str, Any]],
    alias_counts: dict[str, int],
    known_feature_outputs: set[str],
) -> dict[str, Any]:
    inputs = deepcopy(output.get("inputs") or feature.get("inputs") or {})
    build_type = _feature_build_type(feature)

    if build_type == "source_layer" and not inputs:
        source_input = output.get("source") or feature.get("source")
        if not isinstance(source_input, dict):
            raise ConfigValidationError(
                f"Source-layer feature {feature.get('name')!r} is missing source."
            )
        inputs = {"x": source_input}

    if not isinstance(inputs, dict) or not inputs:
        raise ConfigValidationError(f"Feature {feature.get('name')!r} has no inputs.")

    compiled_inputs: dict[str, dict[str, Any]] = {}
    for alias, input_cfg in inputs.items():
        if not isinstance(input_cfg, dict):
            raise ConfigValidationError(
                f"Feature {feature.get('name')!r} input {alias!r} must be a dictionary."
            )
        compiled_inputs[str(alias)] = _compile_feature_input(
            input_cfg=input_cfg,
            source_entries=source_entries,
            source_key_to_alias=source_key_to_alias,
            source_alias_to_entry=source_alias_to_entry,
            alias_counts=alias_counts,
            known_feature_outputs=known_feature_outputs,
            used_by_feature=_feature_output_name(feature, output),
        )
    return compiled_inputs


def _derived_feature_from_feature_output(
    feature: dict[str, Any],
    output: dict[str, Any],
    *,
    source_entries: list[dict[str, Any]],
    source_key_to_alias: dict[str, str],
    source_alias_to_entry: dict[str, dict[str, Any]],
    alias_counts: dict[str, int],
    known_feature_outputs: set[str],
    known_feature_semantics: dict[str, str],
) -> dict[str, Any]:
    build_type = _feature_build_type(feature)
    output_name = _feature_output_name(feature, output)
    inputs = _normalise_feature_inputs(
        feature,
        output,
        source_entries=source_entries,
        source_key_to_alias=source_key_to_alias,
        source_alias_to_entry=source_alias_to_entry,
        alias_counts=alias_counts,
        known_feature_outputs=known_feature_outputs,
    )

    inferred_semantics = _infer_feature_value_semantics(
        feature,
        output,
        known_feature_semantics,
    )
    inferred_dtype = _infer_feature_output_dtype(feature, output, inferred_semantics)

    derived: dict[str, Any] = {
        "name": output_name,
        "description": output.get("description") or feature.get("description"),
        "unit": output.get("unit") or feature.get("unit"),
        "value_semantics": inferred_semantics,
        "output_dtype": inferred_dtype,
        "inputs": inputs,
    }

    for key in ["title", "valid_range", "temporal_meaning"]:
        if output.get(key) is not None or feature.get(key) is not None:
            derived[key] = output.get(key, feature.get(key))

    evaluation_stage = _normalise_feature_evaluation_stage(
        output.get("evaluation_stage", feature.get("evaluation_stage"))
    )
    if evaluation_stage:
        derived["evaluation_stage"] = evaluation_stage
        if evaluation_stage == "native_then_resample":
            derived["evaluation_resolution_m"] = _normalise_feature_evaluation_resolution(
                output.get("evaluation_resolution_m", feature.get("evaluation_resolution_m"))
            )
            post_resampling = _normalise_feature_post_resampling(
                output.get("post_resampling", feature.get("post_resampling"))
                or output.get("final_resampling", feature.get("final_resampling"))
            )
            if post_resampling:
                derived["post_resampling"] = post_resampling

    pre_resampling = output.get(
        "pre_resample_input_resampling",
        feature.get("pre_resample_input_resampling"),
    )
    if pre_resampling is not None:
        derived["pre_resample_input_resampling"] = _normalise_feature_post_resampling(pre_resampling)

    if build_type == "source_layer":
        derived.update({"operation": "expression", "expression": "x"})
    elif build_type == "expression":
        expression = output.get("expression") or feature.get("expression")
        if not expression:
            raise ConfigValidationError(f"Expression feature {output_name!r} is missing expression.")
        derived.update({"operation": "expression", "expression": str(expression)})
    elif build_type in {"recipe", "masking"}:
        recipe = output.get("recipe") or feature.get("recipe")
        if not recipe:
            raise ConfigValidationError(f"Recipe feature {output_name!r} is missing recipe.")
        derived.update(
            {
                "operation": "recipe",
                "recipe": str(recipe),
                "parameters": deepcopy(output.get("parameters") or feature.get("parameters") or {}),
            }
        )
    elif build_type in {"terrain", "focal", "distance", "spatial"}:
        operation = str(output.get("operation") or feature.get("operation") or "")
        if build_type != "spatial":
            operation = build_type
        if operation not in {"terrain", "focal", "distance"}:
            raise ConfigValidationError(
                f"Spatial feature {output_name!r} must use terrain, focal or distance."
            )
        method = output.get("method") or feature.get("method")
        derived.update(
            {
                "operation": operation,
                "parameters": deepcopy(output.get("parameters") or feature.get("parameters") or {}),
            }
        )
        if method:
            derived["method"] = str(method)
    else:
        raise ConfigValidationError(f"Unsupported feature build_type: {build_type}")

    return {key: value for key, value in derived.items() if value is not None}


def _compile_feature_run_config(run_cfg: dict[str, Any]) -> dict[str, Any]:
    if "sources" in run_cfg or "derived_features" in run_cfg:
        raise ConfigValidationError(
            "Feature-oriented run configs must not include top-level 'sources' "
            "or 'derived_features'. Define final outputs under 'features'."
        )

    features = run_cfg.get("features")
    if not isinstance(features, list) or not features:
        raise ConfigValidationError("Feature-oriented run config requires non-empty 'features'.")

    compiled = deepcopy(run_cfg)
    compiled["_compiled_from_features"] = True
    compiled_sources: list[dict[str, Any]] = []
    compiled_derived: list[dict[str, Any]] = []
    source_key_to_alias: dict[str, str] = {}
    source_alias_to_entry: dict[str, dict[str, Any]] = {}
    alias_counts: dict[str, int] = {}
    known_outputs: set[str] = set()
    known_output_semantics: dict[str, str] = {}
    warnings: list[str] = []

    for feature in features:
        if not isinstance(feature, dict):
            raise ConfigValidationError("Each feature must be a dictionary.")

        for output in _feature_outputs(feature):
            derived = _derived_feature_from_feature_output(
                feature,
                output,
                source_entries=compiled_sources,
                source_key_to_alias=source_key_to_alias,
                source_alias_to_entry=source_alias_to_entry,
                alias_counts=alias_counts,
                known_feature_outputs=known_outputs,
                known_feature_semantics=known_output_semantics,
            )
            if derived["name"] in known_outputs:
                raise ConfigValidationError(
                    f"Duplicate final feature output name: {derived['name']}"
                )
            if len(compiled_derived) >= 500:
                warnings.append(
                    "This run expands to more than 500 final features; it may be slow "
                    "and storage-heavy."
                )
            compiled_derived.append(derived)
            known_outputs.add(str(derived["name"]))
            if derived.get("value_semantics"):
                known_output_semantics[str(derived["name"])] = str(derived["value_semantics"])

    if not compiled_sources:
        raise ConfigValidationError("At least one final feature must depend on a source input.")

    compiled["sources"] = compiled_sources
    compiled["derived_features"] = compiled_derived
    compiled["_feature_compile_warnings"] = sorted(set(warnings))
    return compiled


def compile_run_config(
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Compile run-level convenience blocks that are independent of source loading.
    """
    if run_cfg.get("_compiled_from_features"):
        return deepcopy(run_cfg)

    if "features" not in run_cfg:
        raise ConfigValidationError(
            "Feature-oriented run config requires top-level 'features'. Legacy "
            "'sources'/'derived_features' configs are no longer supported."
        )

    return _compile_feature_run_config(run_cfg)


def _source_stages(
    compiled_run: dict[str, Any],
    source_entry: dict[str, Any],
) -> list[str]:
    default_stages = normalize_stages(compiled_run["run"].get("stages", ["build"]))
    return normalize_stages(source_entry.get("stages", default_stages))


def _stages_need(stages: list[str], stage: str) -> bool:
    return stage in stages


def _compiled_run_requests_build(compiled_run: dict[str, Any]) -> bool:
    for source_entry in compiled_run.get("sources", []) or []:
        if _stages_need(_source_stages(compiled_run, source_entry), "build"):
            return True
    return False


def _validate_derived_runtime_contract(
    compiled_run: dict[str, Any],
    errors: list[str],
) -> None:
    derived = compiled_run.get("derived_features", []) or []
    if not derived:
        return
    if not _compiled_run_requests_build(compiled_run):
        return

    outputs = compiled_run.get("outputs", {}) or {}
    run = compiled_run.get("run", {}) or {}

    if outputs.get("copy_rasters", True) is False:
        errors.append("derived_features require outputs.copy_rasters=true.")
    if outputs.get("write_manifest", True) is False:
        errors.append("derived_features require outputs.write_manifest=true.")
    if not run.get("aoi_config"):
        errors.append("derived_features require run.aoi_config.")
    if run.get("resolution_m") is None:
        errors.append("derived_features require run.resolution_m.")


def _load_effective_project_cfg(
    compiled_run: dict[str, Any],
    run_config_path: str | Path | None,
) -> dict[str, Any]:
    base_path = Path(run_config_path) if run_config_path is not None else None
    project_path = resolve_path(
        compiled_run["run"].get("project_config", "configs/project.yaml"),
        base_path=base_path,
        must_exist=True,
    )
    project_cfg = load_yaml(project_path)
    project_cfg["_config_path"] = str(project_path)
    return apply_run_overrides_to_project_cfg(project_cfg, compiled_run)


def _resolve_domain_config(
    source_cfg: dict[str, Any],
    key: str,
) -> Path:
    domains = source_cfg.get("domains", {}) or {}
    if key not in domains:
        raise ConfigValidationError(f"Missing domains.{key}.")
    return resolve_path(
        domains[key],
        base_path=source_cfg.get("_config_path"),
        must_exist=True,
    )


def _validate_source_runtime_contract(
    *,
    source_id: str,
    source_cfg: dict[str, Any],
    stages: list[str],
    project_cfg: dict[str, Any] | None,
    warnings: list[str],
    seen_grid_paths: set[str],
) -> None:
    if _stages_need(stages, "clip") or _stages_need(stages, "build"):
        _resolve_domain_config(source_cfg, "clip_aoi_config")

    if not _stages_need(stages, "build"):
        return

    output_aoi_path = _resolve_domain_config(source_cfg, "output_aoi_config")
    if project_cfg is None:
        return

    output_aoi_cfg = load_yaml(output_aoi_path)
    target_resolution_m = int(source_cfg["processing"]["target_resolution_m"])
    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    key = str(grid_path)
    if grid_path.exists() or key in seen_grid_paths:
        return

    seen_grid_paths.add(key)
    warnings.append(
        f"{source_id}: build stage requires target grid that does not exist yet: "
        f"{grid_path}. Create it with pirineus-raster make-grid before running build."
    )


def validate_researcher_run_config(
    run_cfg: dict[str, Any],
    run_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate feature-oriented run configs without touching raster data.
    """
    errors: list[str] = []
    warnings: list[str] = []
    source_summaries: list[dict[str, Any]] = []

    try:
        compiled_run = compile_run_config(run_cfg)
        warnings.extend(compiled_run.get("_feature_compile_warnings", []) or [])
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
    seen_grid_paths: set[str] = set()

    try:
        project_cfg = _load_effective_project_cfg(
            compiled_run=compiled_run,
            run_config_path=run_config_path,
        )
    except Exception as exc:
        project_cfg = None
        errors.append(f"project_config: {exc}")

    _validate_derived_runtime_contract(compiled_run, errors)

    for source_entry in compiled_run.get("sources", []):
        source_id = source_entry.get("id") or source_entry.get("config")
        try:
            source_config_path = resolve_path(
                source_entry["config"],
                base_path=base_path,
                must_exist=True,
            )
            source_cfg = load_yaml(source_config_path)
            source_cfg["_config_path"] = str(source_config_path)
            source_cfg = normalize_source_domains(source_cfg)
            source_cfg = apply_run_overrides_to_source_cfg(
                source_cfg=source_cfg,
                run_cfg=compiled_run,
            )
            source_cfg = expand_source_config(source_cfg)
            compiled_source = compile_source_config_for_run(source_cfg, source_entry)
            compiled_source = expand_source_config(compiled_source)
            source_summary = _source_summary(source_entry, compiled_source)
            source_summary["config"] = str(source_config_path)
            source_summaries.append(source_summary)

            source_stages = _source_stages(compiled_run, source_entry)
            _validate_source_runtime_contract(
                source_id=str(source_summary["id"]),
                source_cfg=compiled_source,
                stages=source_stages,
                project_cfg=project_cfg,
                warnings=warnings,
                seen_grid_paths=seen_grid_paths,
            )

            if (
                source_summary.get("temporal_output_mode") == "raw_slices"
                and source_summary.get("estimated_layers", 0) > 500
            ):
                warnings.append(
                    f"{source_summary['id']}: raw_slices will generate "
                    f"{source_summary['estimated_layers']} rasters. "
                    "This is valid but can be slow and storage-heavy."
                )
        except Exception as exc:
            used_by = source_entry.get("used_by_features", []) or []
            errors.append(_format_source_validation_error(source_id, used_by, exc))

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


def _enabled_output_keys(collection: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in collection.items()
        if (
            isinstance(value, dict)
            and bool(value.get("enabled", False))
            and bool(value.get("build_output_enabled", True))
        )
    ]


def _range_count(value: Any, default: int = 1) -> int:
    if not value:
        return default
    values = [int(item) for item in _as_list(value)]
    if len(values) == 2:
        return max(0, values[1] - values[0] + 1)
    return len(values)


def _estimate_pdca_layers(
    source_cfg: dict[str, Any],
    enabled_variables: list[str],
) -> int:
    expected = source_cfg.get("dataset", {}).get("expected_variables", []) or []
    temporal_cfg = source_cfg.get("temporal", {}) or {}
    selected_layers = temporal_cfg.get("layers", {}) or {}

    month_filter = set(selected_layers.get("months", []) or [])
    season_filter = set(selected_layers.get("seasons", []) or [])
    annual_filter = selected_layers.get("annual")
    annual_index_filter = selected_layers.get("annual_index")
    has_filter = bool(selected_layers)

    total = 0
    enabled = set(enabled_variables)

    for item in expected:
        variable_key = item.get("variable_key")
        if variable_key not in enabled:
            continue

        for layer in item.get("temporal_layers", []) or []:
            if layer is None:
                if not has_filter or annual_index_filter is not False:
                    total += 1
                continue

            layer_name = str(layer).lower()
            if layer_name == "annual":
                if not has_filter or annual_filter is not False:
                    total += 1
            elif layer_name in MONTH_NAMES:
                if not has_filter or layer_name in month_filter:
                    total += 1
            elif layer_name in SEASON_NAMES:
                if not has_filter or layer_name in season_filter:
                    total += 1

    return total


def _yearly_summary_base_name(source_cfg: dict[str, Any], variable_cfg: dict[str, Any], variable: str) -> str:
    base = str(variable_cfg.get("generated_from_group", variable))
    context = variable_cfg.get("generation_context", {}) or {}
    for context_key in (source_cfg.get("dimension_context_keys", {}) or {}).values():
        value = context.get(context_key)
        if value is not None:
            base = f"{base}_{value}"
    return base


def _yearly_summary_output_names(
    source_cfg: dict[str, Any],
    variables: list[str],
) -> set[str]:
    all_variables = source_cfg.get("variables", {}) or {}
    return {
        _yearly_summary_base_name(source_cfg, all_variables[variable], variable)
        for variable in variables
        if variable in all_variables and isinstance(all_variables[variable], dict)
    }


def _yearly_summary_applicable_names(
    source_cfg: dict[str, Any],
    enabled_names: set[str],
    aggregation_variables: list[str],
) -> set[str]:
    if not aggregation_variables:
        return enabled_names

    selected = set(aggregation_variables)
    if selected & enabled_names:
        return selected & enabled_names

    all_variables = source_cfg.get("variables", {}) or {}
    matched: set[str] = set()
    for variable, variable_cfg in all_variables.items():
        if not isinstance(variable_cfg, dict):
            continue
        group = str(variable_cfg.get("generated_from_group", variable))
        base = _yearly_summary_base_name(source_cfg, variable_cfg, variable)
        if group in selected or variable in selected:
            matched.add(base)
    return matched & enabled_names


def _source_summary(
    source_entry: dict[str, Any],
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    source = source_cfg.get("source", {}) or {}
    variables = _enabled_output_keys(source_cfg.get("variables", {}) or {})
    indices = _enabled_keys(source_cfg.get("indices", {}) or {})
    category_fractions = source_cfg.get("category_fractions", []) or []

    layer_count = len(variables) + len(indices) + len(category_fractions)

    dimensions = {
        key: source_cfg.get(key, [])
        for key in ["gcms", "ssps", "periods"]
        if source_cfg.get(key)
    }
    dataset = source_cfg.get("dataset", {}) or {}
    layer_structure = dataset.get("layer_structure")
    temporal_cfg = source_cfg.get("temporal", {}) or {}
    output_mode = temporal_cfg.get("output_mode")
    output_dimension_keys = (
        ["gcms", "ssps", "periods"]
        if layer_structure == "future_monthly_multiband"
        else []
    )

    for key in output_dimension_keys:
        values = dimensions.get(key, [])
        layer_count *= max(1, len(values))

    aggregations = source_cfg.get("temporal_aggregations", []) or []
    if output_mode == "raw_slices":
        raw_cfg = temporal_cfg.get("raw_slices", {}) or {}
        month_count = _range_count(raw_cfg.get("months"), default=12)
        year_count = (
            _range_count(raw_cfg.get("years"), default=1)
            if layer_structure == "monthly_time_series"
            else 1
        )
        layer_count = (len(variables) + len(indices)) * month_count * year_count
        for key in output_dimension_keys:
            values = dimensions.get(key, [])
            layer_count *= max(1, len(values))
    elif layer_structure == "pdca_nested_zip_geotiff_collection":
        layer_count = _estimate_pdca_layers(source_cfg, variables)
    elif aggregations:
        layer_count = 0
        enabled_names = set(variables) | set(indices)
        if layer_structure == "yearly_static_collection":
            enabled_names = _yearly_summary_output_names(source_cfg, variables)
        for aggregation in aggregations:
            aggregation_variables = aggregation.get("variables") or sorted(enabled_names)
            if layer_structure == "yearly_static_collection":
                applicable = _yearly_summary_applicable_names(
                    source_cfg,
                    enabled_names,
                    [str(item) for item in aggregation_variables],
                )
            else:
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
        "temporal_output_mode": output_mode,
        "aggregations": [item.get("name") for item in aggregations],
        "category_fractions": [item.get("name") for item in category_fractions],
        "vector_layers": vector_layers,
        "estimated_layers": layer_count,
        "used_by_features": sorted(set(source_entry.get("used_by_features", []) or [])),
    }


def load_and_compile_run_config(
    run_config_path: str | Path,
) -> dict[str, Any]:
    path = resolve_path(run_config_path, must_exist=True)
    cfg = load_yaml(path)
    return compile_run_config(cfg)


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def render_run_config_yaml(
    run_cfg: dict[str, Any],
    compile_groups: bool = True,
) -> str:
    cfg = deepcopy(run_cfg)
    for key in [
        "_compiled_from_features",
        "_feature_compile_warnings",
        "sources",
        "derived_features",
    ]:
        if key.startswith("_") or "features" in cfg:
            cfg.pop(key, None)
    return yaml.dump(
        cfg,
        Dumper=_NoAliasSafeDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
