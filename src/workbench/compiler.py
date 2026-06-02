from __future__ import annotations

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

    for name in selected:
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

    for key, selected_values in dimensions_cfg.items():
        selected = [str(item) for item in _as_list(selected_values)]

        available = cfg.get(key)
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

        cfg[key] = selected


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
    compiled: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for item in selected:
        if not isinstance(item, dict):
            raise ConfigValidationError("Each category fraction must be a dictionary.")

        variable = str(item.get("variable", ""))
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

            build_output_enabled = bool(variable_cfg.get("enabled", False))
            variable_cfg["enabled"] = True
            variable_cfg.setdefault("build_output_enabled", build_output_enabled)

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
        _validate_range_pair(
            aggregation["years"],
            name=f"Aggregation {aggregation['name']!r} years",
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
    unknown = sorted(set(selected_variables) - known_variables)
    if unknown:
        raise ConfigValidationError(
            f"Aggregation {aggregation['name']!r} references unknown variables: {unknown}"
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
    base = {
        "filename": f"{name}.tif",
        "method": method,
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
        years = _validate_range_pair(
            aggregation["years"],
            name=f"Aggregation {aggregation['name']!r} years",
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
        enabled = variable in selected_names and year in selected_years
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


def compile_run_config(
    run_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Compile run-level convenience blocks that are independent of source loading.
    """
    return expand_derived_feature_groups(run_cfg)


def _source_stages(
    compiled_run: dict[str, Any],
    source_entry: dict[str, Any],
) -> list[str]:
    default_stages = normalize_stages(compiled_run["run"].get("stages", ["build"]))
    return normalize_stages(source_entry.get("stages", default_stages))


def _stages_need(stages: list[str], stage: str) -> bool:
    return stage in stages


def _validate_derived_runtime_contract(
    compiled_run: dict[str, Any],
    errors: list[str],
) -> None:
    derived = compiled_run.get("derived_features", []) or []
    if not derived:
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
            enabled_names = {
                str(source_cfg["variables"][variable].get("generated_from_group", variable))
                for variable in variables
                if variable in source_cfg.get("variables", {})
            }
        for aggregation in aggregations:
            aggregation_variables = aggregation.get("variables") or sorted(enabled_names)
            if layer_structure == "yearly_static_collection":
                aggregation_names = {
                    str(
                        source_cfg.get("variables", {})
                        .get(str(variable), {})
                        .get("generated_from_group", variable)
                    )
                    for variable in aggregation_variables
                }
            else:
                aggregation_names = set(aggregation_variables)
            applicable = aggregation_names & enabled_names
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
