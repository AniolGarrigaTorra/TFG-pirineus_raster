from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


ALLOWED_BINOPS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
}

ALLOWED_UNARYOPS = {
    ast.UAdd,
    ast.USub,
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
}


def read_raster_array_as_nan(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a single-band raster as float32 and convert nodata to np.nan.

    Returns:
      array, profile
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)

    array[~np.isfinite(array)] = np.nan

    return array, profile


def _validate_expression_node(node: ast.AST, allowed_names: set[str]) -> None:
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

    env: dict[str, Any] = {}
    env.update(ALLOWED_FUNCTIONS)
    env.update(variables)

    result = eval(
        compile(tree, filename="<raster-expression>", mode="eval"),
        {"__builtins__": {}},
        env,
    )

    result = np.asarray(result, dtype=np.float32)
    result[~np.isfinite(result)] = np.nan

    return result