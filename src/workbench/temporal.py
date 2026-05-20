from __future__ import annotations

from typing import Any


MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

SEASON_NAMES = ["winter", "spring", "summer", "autumn"]


def _period_year_bounds(periods: list[Any]) -> list[int] | None:
    years: list[int] = []

    for period in periods:
        text = str(period)
        if "-" not in text:
            continue
        left, right = text.split("-", 1)
        try:
            years.extend([int(left), int(right)])
        except ValueError:
            continue

    if not years:
        return None

    return [min(years), max(years)]


def _pdca_temporal_layers(source_cfg: dict[str, Any]) -> dict[str, Any]:
    expected = source_cfg.get("dataset", {}).get("expected_variables", []) or []
    months: set[str] = set()
    seasons: set[str] = set()
    has_annual = False
    has_annual_index = False

    for item in expected:
        for layer in item.get("temporal_layers", []) or []:
            if layer is None:
                has_annual_index = True
                continue

            layer_name = str(layer).lower()
            if layer_name == "annual":
                has_annual = True
            elif layer_name in MONTH_NAMES:
                months.add(layer_name)
            elif layer_name in SEASON_NAMES:
                seasons.add(layer_name)

    return {
        "annual": has_annual,
        "annual_index": has_annual_index,
        "months": [item for item in MONTH_NAMES if item in months],
        "seasons": [item for item in SEASON_NAMES if item in seasons],
    }


def _temporal_postprocess_outputs(source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    temporal_cfg = source_cfg.get("temporal_postprocess", {}) or {}
    outputs = temporal_cfg.get("output_variables", {}) or {}
    items: list[dict[str, Any]] = []

    for name, cfg in outputs.items():
        items.append(
            {
                "name": name,
                "method": cfg.get("method"),
                "months": cfg.get("months"),
                "threshold": cfg.get("threshold"),
                "comparison": cfg.get("comparison"),
                "unit": cfg.get("unit"),
                "description": cfg.get("description"),
            }
        )

    return items


def infer_temporal_capability(source_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = source_cfg.get("dataset", {}) or {}
    layer_structure = dataset.get("layer_structure")
    temporal_axis = dataset.get("temporal_axis")

    if layer_structure in {
        "static_single",
        "static_multi",
        "static_index_set",
        "vector_categorical",
    }:
        return {
            "kind": "static",
            "label": "Static layers",
            "temporal_axis": None,
            "aggregation_stage": "none",
            "default_output_mode": "static",
            "output_modes": ["static"],
            "aggregation_forms": [],
            "supports_custom_aggregations": False,
            "supports_raw_slices": False,
        }

    if layer_structure == "monthly_climatology":
        return {
            "kind": "monthly_climatology",
            "label": "Monthly climatology",
            "temporal_axis": temporal_axis or "month",
            "aggregation_stage": "build",
            "default_output_mode": "aggregate",
            "output_modes": ["aggregate", "raw_slices"],
            "aggregation_forms": ["month_range_metric"],
            "supports_custom_aggregations": True,
            "supports_raw_slices": True,
            "default_months": [1, 12],
        }

    if layer_structure == "future_monthly_multiband":
        return {
            "kind": "future_monthly",
            "label": "Future monthly climatology",
            "temporal_axis": temporal_axis or "month",
            "aggregation_stage": "build",
            "default_output_mode": "aggregate",
            "output_modes": ["aggregate", "raw_slices"],
            "aggregation_forms": ["month_range_metric"],
            "supports_custom_aggregations": True,
            "supports_raw_slices": True,
            "default_months": [1, 12],
            "dimensioned_by": ["gcms", "ssps", "periods"],
        }

    if layer_structure == "monthly_time_series":
        year_range = _period_year_bounds(source_cfg.get("periods", []) or [])
        return {
            "kind": "year_month_series",
            "label": "Year-month time series",
            "temporal_axis": temporal_axis or "year_month",
            "aggregation_stage": "build",
            "default_output_mode": "aggregate",
            "output_modes": ["aggregate", "raw_slices"],
            "aggregation_forms": [
                "year_range_month_range_metric",
                "year_then_across_years",
            ],
            "supports_custom_aggregations": True,
            "supports_raw_slices": True,
            "default_months": [1, 12],
            "available_years": year_range,
            "default_years": [1991, 2020] if year_range else None,
        }

    if layer_structure == "pdca_nested_zip_geotiff_collection":
        return {
            "kind": "supplied_temporal_collection",
            "label": "Supplied annual/monthly/seasonal layers",
            "temporal_axis": temporal_axis or "supplied_layers",
            "aggregation_stage": "none",
            "default_output_mode": "supplied_layers",
            "output_modes": ["supplied_layers"],
            "aggregation_forms": [],
            "supports_custom_aggregations": False,
            "supports_raw_slices": False,
            "temporal_layers": _pdca_temporal_layers(source_cfg),
        }

    if layer_structure == "temporal_aggregation":
        temporal_cfg = source_cfg.get("temporal_postprocess", {}) or {}
        export_timesteps = temporal_cfg.get("export_timesteps", {}) or {}
        return {
            "kind": "temporal_postprocess",
            "label": "Temporal postprocess",
            "temporal_axis": temporal_axis or "date",
            "aggregation_stage": "download_postprocess",
            "default_output_mode": "postprocess_aggregate",
            "output_modes": ["postprocess_aggregate"],
            "aggregation_forms": [
                "explicit_month_list_metric",
                "threshold_count",
                "valid_observation_count",
            ],
            "supports_custom_aggregations": False,
            "supports_raw_slices": bool(export_timesteps.get("enabled", False)),
            "raw_timesteps_implemented": False,
            "postprocess_outputs": _temporal_postprocess_outputs(source_cfg),
        }

    return {
        "kind": "unknown",
        "label": "Unknown temporal structure",
        "temporal_axis": temporal_axis,
        "aggregation_stage": "unknown",
        "default_output_mode": "static",
        "output_modes": ["static"],
        "aggregation_forms": [],
        "supports_custom_aggregations": False,
        "supports_raw_slices": False,
    }
