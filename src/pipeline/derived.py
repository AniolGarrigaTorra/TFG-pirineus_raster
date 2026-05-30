from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.pipeline.layers import (
    LayerSpec,
    build_layer_catalog_from_manifest,
)
from src.pipeline.raster_ops import (
    load_grid_context,
    read_raster_array_as_nan,
    write_feature_raster,
)


# =============================================================================
# Safe raster expression engine
# =============================================================================


ALLOWED_BINOPS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
}

ALLOWED_CMPOPS = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}

ALLOWED_BOOLOPS = {
    ast.And: np.logical_and,
    ast.Or: np.logical_or,
}

ALLOWED_UNARYOPS = {
    ast.UAdd,
    ast.USub,
    ast.Not,
}

ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "clip": np.clip,
    "isfinite": np.isfinite,
}

NUMERIC_VALUE_SEMANTICS = {
    "intensive",
    "intensive_depth",
    "percentage",
    "fraction",
    "extensive",
    "count",
    None,
}

CATEGORICAL_VALUE_SEMANTICS = {
    "categorical",
    "ordinal",
}

DERIVED_OPERATION_GROUPS = {
    "pixelwise_ops": [
        "expression",
        "recipe",
    ],
    "recipe_ops": [
        "thermal_range",
        "water_balance",
        "aridity_index",
        "seasonal_contrast",
        "snow_persistence_ratio",
        "binary_threshold_mask",
        "class_mask",
        "reclassification",
    ],
    "terrain_ops": [
        "slope",
        "aspect",
        "ruggedness",
        "tpi",
        "roughness",
    ],
    "focal_ops": [
        "mean",
        "std",
        "min",
        "max",
        "sum",
        "majority",
        "diversity",
    ],
    "distance_ops": [
        "distance_to_mask",
        "distance_to_class",
    ],
    "interpolation_ops": [
        "idw",
        "ordinary_kriging",
        "universal_kriging",
        "regression_kriging",
        "thin_plate_spline",
    ],
}


def _validate_expression_node(
    node: ast.AST,
    allowed_names: set[str],
) -> None:
    """
    Validate that an AST expression only contains safe mathematical operations.
    """
    if isinstance(node, ast.Expression):
        _validate_expression_node(node.body, allowed_names)
        return

    if isinstance(node, ast.BinOp):
        if type(node.op) not in ALLOWED_BINOPS:
            raise ValueError(f"Operator not allowed: {type(node.op).__name__}")

        _validate_expression_node(node.left, allowed_names)
        _validate_expression_node(node.right, allowed_names)
        return

    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator not allowed: {type(node.op).__name__}")

        _validate_expression_node(node.operand, allowed_names)
        return

    if isinstance(node, ast.BoolOp):
        if type(node.op) not in ALLOWED_BOOLOPS:
            raise ValueError(f"Boolean operator not allowed: {type(node.op).__name__}")
        for value in node.values:
            _validate_expression_node(value, allowed_names)
        return

    if isinstance(node, ast.Compare):
        _validate_expression_node(node.left, allowed_names)
        for op in node.ops:
            if type(op) not in ALLOWED_CMPOPS:
                raise ValueError(f"Comparison operator not allowed: {type(op).__name__}")
        for comparator in node.comparators:
            _validate_expression_node(comparator, allowed_names)
        return

    if isinstance(node, ast.Name):
        if node.id not in allowed_names and node.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Unknown name in expression: {node.id}")
        return

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants are allowed: {node.value!r}")
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")

        if node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: {node.func.id}")

        for arg in node.args:
            _validate_expression_node(arg, allowed_names)

        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in expressions.")

        return

    raise ValueError(f"Expression element not allowed: {type(node).__name__}")


def _eval_expression_node(
    node: ast.AST,
    env: dict[str, Any],
) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body, env)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return env[node.id]

    if isinstance(node, ast.UnaryOp):
        value = _eval_expression_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return np.logical_not(value)

    if isinstance(node, ast.BinOp):
        left = _eval_expression_node(node.left, env)
        right = _eval_expression_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            with np.errstate(divide="ignore", invalid="ignore"):
                return left / right
        if isinstance(node.op, ast.Pow):
            with np.errstate(invalid="ignore"):
                return left ** right

    if isinstance(node, ast.BoolOp):
        values = [_eval_expression_node(value, env) for value in node.values]
        func = ALLOWED_BOOLOPS[type(node.op)]
        result = values[0]
        for value in values[1:]:
            result = func(result, value)
        return result

    if isinstance(node, ast.Compare):
        left = _eval_expression_node(node.left, env)
        comparisons = []
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_expression_node(comparator, env)
            comparisons.append(ALLOWED_CMPOPS[type(op)](left, right))
            left = right
        result = comparisons[0]
        for comparison in comparisons[1:]:
            result = np.logical_and(result, comparison)
        return result

    if isinstance(node, ast.Call):
        func = ALLOWED_FUNCTIONS[node.func.id]
        args = [_eval_expression_node(arg, env) for arg in node.args]
        return func(*args)

    raise ValueError(f"Expression element not executable: {type(node).__name__}")


def evaluate_raster_expression(
    expression: str,
    variables: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Safely evaluate a mathematical expression over raster arrays.

    Example:
      expression = "tmax - tmin"
      variables = {"tmax": arr1, "tmin": arr2}
    """
    if not variables:
        raise ValueError("No variables provided for expression evaluation.")

    shapes = {array.shape for array in variables.values()}
    if len(shapes) != 1:
        raise ValueError(f"Input rasters do not share the same shape: {shapes}")

    tree = ast.parse(expression, mode="eval")
    _validate_expression_node(tree, allowed_names=set(variables))

    env: dict[str, Any] = dict(variables)
    env.update(ALLOWED_FUNCTIONS)

    result = _eval_expression_node(tree, env)

    result = np.asarray(result, dtype=np.float32)
    if result.shape == ():
        first_shape = next(iter(variables.values())).shape
        result = np.full(first_shape, float(result), dtype=np.float32)
    result[~np.isfinite(result)] = np.nan

    return result


# =============================================================================
# Derived operation registries
# =============================================================================


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32)


def _recipe_expression(
    recipe: str,
    parameters: dict[str, Any],
) -> str:
    convention = str(parameters.get("convention", "prec_over_pet"))
    metric = str(parameters.get("metric", "difference"))

    recipes = {
        "thermal_range": "tmax - tmin",
        "water_balance": "prec - pet",
        "snow_persistence_ratio": "snow_days / valid_days",
    }

    if recipe == "aridity_index":
        if convention == "pet_over_prec":
            return "pet / prec"
        return "prec / pet"

    if recipe == "seasonal_contrast":
        if metric == "ratio":
            return "a / b"
        return "a - b"

    if recipe == "binary_threshold_mask":
        operator = str(parameters.get("operator", ">="))
        threshold = float(parameters.get("threshold", 0))
        if operator not in {">", ">=", "<", "<=", "==", "!="}:
            raise ValueError(f"Unsupported threshold operator: {operator}")
        return f"where(x {operator} {threshold}, 1, 0)"

    if recipe == "class_mask":
        class_value = float(parameters.get("class_value"))
        return f"where(x == {class_value}, 1, 0)"

    if recipe == "reclassification":
        classes = parameters.get("classes", {}) or {}
        if not isinstance(classes, dict) or not classes:
            raise ValueError("reclassification recipe requires parameters.classes.")
        expression = "x"
        for raw_value, new_value in classes.items():
            expression = f"where(x == {float(raw_value)}, {float(new_value)}, {expression})"
        return expression

    if recipe not in recipes:
        raise ValueError(f"Unsupported derived recipe: {recipe}")

    return recipes[recipe]


def _terrain_slope(
    array: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(array.astype(np.float32), pixel_size, pixel_size)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope[~np.isfinite(slope)] = np.nan
    return slope.astype(np.float32)


def _terrain_aspect(
    array: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(array.astype(np.float32), pixel_size, pixel_size)
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))
    aspect = np.mod(aspect, 360.0)
    aspect[~np.isfinite(aspect)] = np.nan
    return aspect.astype(np.float32)


def _window_stat(
    array: np.ndarray,
    radius: int,
    method: str,
) -> np.ndarray:
    if radius < 1:
        raise ValueError("Focal radius must be >= 1.")

    size = radius * 2 + 1
    padded = np.pad(
        array.astype(np.float32),
        radius,
        mode="constant",
        constant_values=np.nan,
    )
    windows = sliding_window_view(padded, (size, size))

    with np.errstate(invalid="ignore"):
        if method == "mean":
            return np.nanmean(windows, axis=(-2, -1)).astype(np.float32)
        if method == "std":
            return np.nanstd(windows, axis=(-2, -1)).astype(np.float32)
        if method == "min":
            return np.nanmin(windows, axis=(-2, -1)).astype(np.float32)
        if method == "max":
            return np.nanmax(windows, axis=(-2, -1)).astype(np.float32)
        if method == "sum":
            return np.nansum(windows, axis=(-2, -1)).astype(np.float32)
        if method == "roughness":
            return (np.nanmax(windows, axis=(-2, -1)) - np.nanmin(windows, axis=(-2, -1))).astype(np.float32)
        if method == "tpi":
            center = array.astype(np.float32)
            mean = np.nanmean(windows, axis=(-2, -1))
            return (center - mean).astype(np.float32)
        if method == "ruggedness":
            center = array.astype(np.float32)
            diff = windows - center[..., None, None]
            return np.sqrt(np.nanmean(diff**2, axis=(-2, -1))).astype(np.float32)

    if method in {"majority", "diversity"}:
        return _categorical_window_stat(windows, method)

    raise ValueError(f"Unsupported focal method: {method}")


def _categorical_window_stat(
    windows: np.ndarray,
    method: str,
) -> np.ndarray:
    height, width = windows.shape[:2]
    result = np.full((height, width), np.nan, dtype=np.float32)

    for row in range(height):
        for col in range(width):
            values = windows[row, col]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            unique, counts = np.unique(values, return_counts=True)
            if method == "majority":
                result[row, col] = unique[np.argmax(counts)]
            else:
                result[row, col] = float(unique.size)

    return result


def _distance_to_mask(
    mask: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    try:
        from scipy import ndimage  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "distance derived operations require scipy. Install scipy or avoid "
            "operation=distance for this run."
        ) from exc

    target = np.isfinite(mask) & (mask > 0)
    distance = ndimage.distance_transform_edt(~target) * float(pixel_size)
    distance[~np.isfinite(mask)] = np.nan
    return distance.astype(np.float32)


def _first_array(
    input_arrays: dict[str, np.ndarray],
    preferred: list[str],
) -> np.ndarray:
    for key in preferred:
        if key in input_arrays:
            return input_arrays[key]
    return next(iter(input_arrays.values()))


def evaluate_derived_operation(
    derived_cfg: dict[str, Any],
    input_arrays: dict[str, np.ndarray],
    grid_resolution_m: int,
) -> tuple[np.ndarray, str, str]:
    operation = str(derived_cfg.get("operation", "expression"))
    parameters = derived_cfg.get("parameters", {}) or {}

    if operation == "expression":
        expression = str(derived_cfg["expression"])
        return (
            evaluate_raster_expression(expression, input_arrays),
            operation,
            expression,
        )

    if operation == "recipe":
        recipe = str(derived_cfg["recipe"])
        expression = _recipe_expression(recipe, parameters)
        return (
            evaluate_raster_expression(expression, input_arrays),
            f"recipe:{recipe}",
            expression,
        )

    if operation == "terrain":
        method = str(derived_cfg.get("method", parameters.get("method", "slope")))
        source = _first_array(input_arrays, ["dem", "x"])
        if method == "slope":
            return _terrain_slope(source, grid_resolution_m), "terrain:slope", "slope(dem)"
        if method == "aspect":
            return _terrain_aspect(source, grid_resolution_m), "terrain:aspect", "aspect(dem)"
        if method in {"ruggedness", "tpi", "roughness"}:
            radius = int(parameters.get("radius", derived_cfg.get("radius", 1)))
            return (
                _window_stat(source, radius=radius, method=method),
                f"terrain:{method}",
                f"{method}(dem, radius={radius})",
            )
        raise ValueError(f"Unsupported terrain method: {method}")

    if operation == "focal":
        method = str(derived_cfg.get("method", parameters.get("method", "mean")))
        radius = int(parameters.get("radius", derived_cfg.get("radius", 1)))
        source = _first_array(input_arrays, ["x"])
        return (
            _window_stat(source, radius=radius, method=method),
            f"focal:{method}",
            f"{method}(x, radius={radius})",
        )

    if operation == "distance":
        source = _first_array(input_arrays, ["mask", "x"])
        class_value = parameters.get("class_value")
        if class_value is not None:
            source = np.where(source == float(class_value), 1.0, 0.0)
        return (
            _distance_to_mask(source, grid_resolution_m),
            "distance:distance_to_mask",
            "distance_to_mask(mask)",
        )

    raise ValueError(f"Unsupported derived operation: {operation}")


# =============================================================================
# Manifest helpers
# =============================================================================


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(
    dataset_dir: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# =============================================================================
# Layer matching
# =============================================================================


def _months_match(
    layer_months: list[int] | None,
    requested_months: list[int] | None,
) -> bool:
    if requested_months is None:
        return True

    if layer_months is None:
        return False

    return list(layer_months) == list(requested_months)


def _matches_layer_query(
    layer: LayerSpec,
    query: dict[str, Any],
) -> bool:
    """
    Return True if a LayerSpec matches an input query from run_config.
    """
    if "provider" in query and layer.provider != query["provider"]:
        return False

    if "product" in query and layer.product != query["product"]:
        return False

    if "source_id" in query and layer.source_id != query["source_id"]:
        return False

    if "variable" in query and layer.variable != query["variable"]:
        return False

    if "aggregation_name" in query and layer.aggregation_name != query["aggregation_name"]:
        return False

    if "aggregation_metric" in query and layer.aggregation_metric != query["aggregation_metric"]:
        return False

    if "months" in query and not _months_match(layer.months, query["months"]):
        return False

    if "gcm" in query and layer.gcm != query["gcm"]:
        return False

    if "ssp" in query and layer.ssp != query["ssp"]:
        return False

    if "period" in query and layer.period != query["period"]:
        return False

    return True


def _find_layer(
    layers: list[LayerSpec],
    query: dict[str, Any],
    input_name: str,
) -> LayerSpec:
    matches = [
        layer
        for layer in layers
        if _matches_layer_query(layer, query)
    ]

    if not matches:
        raise ValueError(
            f"No layer found for derived feature input '{input_name}' "
            f"with query: {query}"
        )

    if len(matches) > 1:
        match_names = [layer.name for layer in matches]
        raise ValueError(
            f"Ambiguous input '{input_name}'. Query matched {len(matches)} layers: "
            f"{match_names}. Add provider/product/source_id/months/period to disambiguate."
        )

    return matches[0]


# =============================================================================
# Derived validation and warnings
# =============================================================================


def _expression_tree(expression: str) -> ast.Expression:
    return ast.parse(expression, mode="eval")


def _expression_has_node(
    expression: str,
    node_types: tuple[type[ast.AST], ...],
) -> bool:
    tree = _expression_tree(expression)
    return any(isinstance(node, node_types) for node in ast.walk(tree))


def _expression_has_division(expression: str) -> bool:
    tree = _expression_tree(expression)
    return any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(tree))


def _layer_metadata_value(layer: LayerSpec, key: str) -> Any:
    if getattr(layer, key, None) is not None:
        return getattr(layer, key)
    return layer.metadata.get(key)


def _layer_value_semantics(layer: LayerSpec) -> str | None:
    value = _layer_metadata_value(layer, "value_semantics")
    return str(value) if value is not None else None


def _layer_native_resolution_m(layer: LayerSpec) -> float | None:
    value = layer.metadata.get("native_resolution_m")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _range_includes_zero(value: Any) -> bool:
    if not value or not isinstance(value, (list, tuple)) or len(value) != 2:
        return True
    low, high = float(value[0]), float(value[1])
    return low <= 0 <= high


def validate_derived_feature_definition(
    derived_cfg: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    operation = str(derived_cfg.get("operation", "expression"))
    inputs_cfg = derived_cfg.get("inputs", {})

    for key in ["name", "inputs"]:
        if key not in derived_cfg:
            raise ValueError(f"Derived feature is missing required key: {key}")

    if not isinstance(inputs_cfg, dict) or not inputs_cfg:
        raise ValueError(f"Derived feature '{derived_cfg.get('name')}' has no inputs.")

    if operation == "expression":
        if "expression" not in derived_cfg:
            raise ValueError(
                f"Derived expression feature '{derived_cfg['name']}' is missing expression."
            )
        tree = _expression_tree(str(derived_cfg["expression"]))
        _validate_expression_node(tree, allowed_names=set(inputs_cfg))
        return warnings

    if operation == "recipe":
        recipe = derived_cfg.get("recipe")
        if not recipe:
            raise ValueError(
                f"Derived recipe feature '{derived_cfg['name']}' is missing recipe."
            )
        expression = _recipe_expression(str(recipe), derived_cfg.get("parameters", {}) or {})
        tree = _expression_tree(expression)
        _validate_expression_node(tree, allowed_names=set(inputs_cfg))
        return warnings

    if operation == "terrain":
        method = str(derived_cfg.get("method", (derived_cfg.get("parameters", {}) or {}).get("method", "slope")))
        if method not in DERIVED_OPERATION_GROUPS["terrain_ops"]:
            raise ValueError(f"Unsupported terrain derived method: {method}")
        return warnings

    if operation == "focal":
        method = str(derived_cfg.get("method", (derived_cfg.get("parameters", {}) or {}).get("method", "mean")))
        if method not in DERIVED_OPERATION_GROUPS["focal_ops"]:
            raise ValueError(f"Unsupported focal derived method: {method}")
        return warnings

    if operation == "distance":
        return warnings

    raise ValueError(f"Unsupported derived operation: {operation}")


def validate_derived_feature_inputs(
    derived_cfg: dict[str, Any],
    input_layers: dict[str, LayerSpec],
    effective_expression: str,
) -> list[str]:
    warnings: list[str] = []
    layers = list(input_layers.values())

    for attr in ["period", "gcm", "ssp", "aggregation_name"]:
        values = sorted({str(getattr(layer, attr)) for layer in layers if getattr(layer, attr) is not None})
        if len(values) > 1:
            warnings.append(
                f"Derived feature mixes different {attr} values: {values}."
            )

    semantics = {
        name: _layer_value_semantics(layer)
        for name, layer in input_layers.items()
    }
    if any(value in CATEGORICAL_VALUE_SEMANTICS for value in semantics.values()):
        arithmetic = _expression_has_node(effective_expression, (ast.BinOp,))
        if arithmetic:
            warnings.append(
                "Derived expression uses arithmetic with categorical/ordinal inputs. "
                "Use masks or reclassification unless this is intentional."
            )

    if _expression_has_division(effective_expression):
        risky = [
            name
            for name, layer in input_layers.items()
            if _range_includes_zero(_layer_metadata_value(layer, "valid_range"))
        ]
        if risky:
            warnings.append(
                "Derived expression contains division and these inputs may include "
                f"zero or unknown ranges: {risky}."
            )

    units = sorted(
        {
            str(layer.unit)
            for layer in layers
            if layer.unit not in [None, "", "class"]
        }
    )
    if len(units) > 1 and _expression_has_node(effective_expression, (ast.Add, ast.Sub)):
        warnings.append(
            f"Derived expression adds/subtracts layers with different units: {units}."
        )

    native_resolutions = [
        value
        for value in (_layer_native_resolution_m(layer) for layer in layers)
        if value is not None and value > 0
    ]
    if len(native_resolutions) > 1:
        ratio = max(native_resolutions) / min(native_resolutions)
        if ratio >= 4:
            warnings.append(
                "Derived feature mixes inputs with strongly different native "
                f"resolutions: {native_resolutions} m."
            )

    return warnings


# =============================================================================
# Derived feature metadata and manifest update
# =============================================================================


def _build_derived_metadata(
    derived_cfg: dict[str, Any],
    input_layers: dict[str, LayerSpec],
    output_path: Path,
    run_name: str,
    operation: str,
    effective_expression: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "provider": "derived",
        "product": "derived_features",
        "source_id": "derived",
        "layer_type": "derived",
        "variable": derived_cfg["name"],
        "variable_description": derived_cfg.get("description"),
        "unit": derived_cfg.get("unit"),
        "data_type": derived_cfg.get("data_type"),
        "value_semantics": derived_cfg.get("value_semantics"),
        "valid_range": derived_cfg.get("valid_range"),
        "operation": operation,
        "operation_type": derived_cfg.get("operation", "expression"),
        "recipe": derived_cfg.get("recipe"),
        "method": derived_cfg.get("method"),
        "parameters": derived_cfg.get("parameters"),
        "expression": derived_cfg.get("expression"),
        "effective_expression": effective_expression,
        "warnings": warnings,
        "temporal_meaning": derived_cfg.get("temporal_meaning"),
        "run_name": run_name,
        "inputs": {
            input_name: {
                "layer_name": layer.name,
                "path": str(layer.path),
                "provider": layer.provider,
                "product": layer.product,
                "source_id": layer.source_id,
                "variable": layer.variable,
                "aggregation_name": layer.aggregation_name,
                "aggregation_metric": layer.aggregation_metric,
                "months": layer.months,
                "year": layer.year,
                "gcm": layer.gcm,
                "ssp": layer.ssp,
                "period": layer.period,
                "unit": layer.unit,
                "valid_range": layer.valid_range,
                "value_semantics": _layer_value_semantics(layer),
                "native_resolution_m": _layer_native_resolution_m(layer),
            }
            for input_name, layer in input_layers.items()
        },
        "output_path": str(output_path),
    }


def _append_derived_raster_to_manifest(
    manifest: dict[str, Any],
    raster_entry: dict[str, Any],
    layer_entry: dict[str, Any],
) -> None:
    manifest.setdefault("rasters", []).append(raster_entry)
    manifest.setdefault("layer_catalog", []).append(layer_entry)

    manifest["n_rasters"] = len(manifest.get("rasters", []))

    layer_summary = manifest.get("layer_summary", {})
    layer_summary["n_layers"] = len(manifest.get("layer_catalog", []))

    providers = sorted(
        {
            layer.get("provider")
            for layer in manifest.get("layer_catalog", [])
            if layer.get("provider")
        }
    )
    products = sorted(
        {
            layer.get("product")
            for layer in manifest.get("layer_catalog", [])
            if layer.get("product")
        }
    )
    variables = sorted(
        {
            layer.get("variable")
            for layer in manifest.get("layer_catalog", [])
            if layer.get("variable")
        }
    )

    layer_summary["providers"] = providers
    layer_summary["products"] = products
    layer_summary["variables"] = variables

    manifest["layer_summary"] = layer_summary


# =============================================================================
# Public API
# =============================================================================


def build_derived_features(
    run_cfg: dict[str, Any],
    project_cfg: dict[str, Any],
    dataset_dir: Path,
    output_aoi_cfg: dict[str, Any],
) -> list[Path]:
    """
    Build derived raster features defined in run_cfg['derived_features'].

    Derived features are evaluated from already generated dataset rasters.
    """
    derived_features = run_cfg.get("derived_features", [])

    if not derived_features:
        return []

    manifest = _load_manifest(dataset_dir)
    layers = build_layer_catalog_from_manifest(manifest)

    rasters_dir = dataset_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_cfg["run"]["name"]
    target_resolution_m = int(run_cfg["run"]["resolution_m"])

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    nodata = float(project_cfg.get("nodata", -9999.0))

    written_paths: list[Path] = []

    for derived_cfg in derived_features:
        validate_derived_feature_definition(derived_cfg)

        name = derived_cfg["name"]
        inputs_cfg = derived_cfg.get("inputs", {})
        operation_name = str(derived_cfg.get("operation", "expression"))

        if not inputs_cfg:
            raise ValueError(f"Derived feature '{name}' has no inputs.")

        print("==============================")
        print(f"[derived] Feature: {name}")
        print(f"[derived] Operation: {operation_name}")

        input_layers: dict[str, LayerSpec] = {}
        input_arrays: dict[str, np.ndarray] = {}

        for input_name, query in inputs_cfg.items():
            layer = _find_layer(
                layers=layers,
                query=query,
                input_name=input_name,
            )

            input_layers[input_name] = layer

            array, _ = read_raster_array_as_nan(layer.path)
            input_arrays[input_name] = array

            print(f"[derived] Input {input_name}: {layer.name}")

        result, operation, effective_expression = evaluate_derived_operation(
            derived_cfg=derived_cfg,
            input_arrays=input_arrays,
            grid_resolution_m=target_resolution_m,
        )
        warnings = validate_derived_feature_inputs(
            derived_cfg=derived_cfg,
            input_layers=input_layers,
            effective_expression=effective_expression,
        )
        for warning in warnings:
            print(f"[derived][warning] {warning}")

        output_name = f"derived_{name}.tif"
        output_path = rasters_dir / output_name

        metadata = _build_derived_metadata(
            derived_cfg=derived_cfg,
            input_layers=input_layers,
            output_path=output_path,
            run_name=run_name,
            operation=operation,
            effective_expression=effective_expression,
            warnings=warnings,
        )

        output_dtype = derived_cfg.get("output_dtype", "float32")

        written_path = write_feature_raster(
            output_path=output_path,
            array=result,
            grid=grid,
            metadata=metadata,
            output_dtype=output_dtype,
            nodata=nodata,
            compression="LZW",
            write_sidecar=True,
            validate=True,
        )

        sidecar_path = written_path.with_suffix(".json")

        raster_entry = {
            "name": written_path.stem,
            "source_id": "derived",
            "original_path": str(written_path),
            "dataset_path": str(written_path),
            "sidecar_json_original_path": str(sidecar_path),
            "sidecar_json_dataset_path": str(sidecar_path),
        }

        layer_entry = {
            "name": written_path.stem,
            "path": str(written_path),
            "provider": "derived",
            "product": "derived_features",
            "source_id": "derived",
            "variable": name,
            "variable_description": derived_cfg.get("description"),
            "unit": derived_cfg.get("unit"),
            "valid_range": derived_cfg.get("valid_range"),
            "aoi": manifest.get("run_aoi_name"),
            "resolution_m": manifest.get("run_resolution_m"),
            "crs": str(grid.crs),
            "nodata": nodata,
            "dtype": output_dtype,
            "layer_type": "derived",
            "sidecar_metadata_path": str(sidecar_path),
            "original_path": str(written_path),
            "dataset_path": str(written_path),
            "metadata": metadata,
        }

        _append_derived_raster_to_manifest(
            manifest=manifest,
            raster_entry=raster_entry,
            layer_entry=layer_entry,
        )

        written_paths.append(written_path)
        print(f"[derived] Written: {written_path}")

    _write_manifest(dataset_dir, manifest)

    return written_paths
