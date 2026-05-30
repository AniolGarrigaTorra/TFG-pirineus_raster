from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

RUNTIME_TEMPLATE_KEYS = {
    "source_resolution",
    "target_resolution_m",
}


class _PartialFormatDict(dict):
    def __missing__(self, key: str) -> str:
        if key in RUNTIME_TEMPLATE_KEYS:
            return "{" + key + "}"
        raise KeyError(key)


def _format_value(value: Any, context: dict[str, Any]) -> Any:
    """
    Recursively format strings inside dict/list/scalar structures using context.

    Special case:
      If a string is exactly "{months}" and context["months"] is a list,
      return the list itself instead of a string representation.
    """
    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}"):
            key = value[1:-1]
            if key in context and not isinstance(context[key], str):
                return deepcopy(context[key])

        try:
            return value.format_map(_PartialFormatDict(context))
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Missing template value {missing!r} while formatting {value!r}. "
                f"Available keys: {sorted(context)}"
            ) from exc

    if isinstance(value, list):
        return [_format_value(item, context) for item in value]

    if isinstance(value, dict):
        return {
            key: _format_value(item, context)
            for key, item in value.items()
        }

    return value


def _normalise_items(group_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build expansion items from either:

    foreach:
      - season: winter
        months: [12, 1, 2]

    or cartesian:
      metric: [mean, max]
      season:
        - season: winter
          months: [12, 1, 2]
        - season: summer
          months: [6, 7, 8]
    """
    if "foreach" in group_cfg:
        items = group_cfg["foreach"]
        if not isinstance(items, list):
            raise TypeError("foreach must be a list.")
        return [dict(item) for item in items]

    if "cartesian" in group_cfg:
        cartesian_cfg = group_cfg["cartesian"]
        if not isinstance(cartesian_cfg, dict):
            raise TypeError("cartesian must be a dict.")

        keys = list(cartesian_cfg)
        value_lists: list[list[Any]] = []

        for key in keys:
            values = cartesian_cfg[key]
            if not isinstance(values, list):
                raise TypeError(f"cartesian.{key} must be a list.")
            value_lists.append(values)

        items: list[dict[str, Any]] = []

        for combination in product(*value_lists):
            item: dict[str, Any] = {}

            for key, value in zip(keys, combination):
                if isinstance(value, dict):
                    item[key] = value
                    item.update(value)
                else:
                    item[key] = value

            items.append(item)

        return items

    raise ValueError("Each group must define either 'foreach' or 'cartesian'.")


def _expand_groups_to_mapping(
    *,
    groups: dict[str, Any],
    target: dict[str, Any],
    target_name: str,
) -> dict[str, Any]:
    """
    Generic group expander.

    Used for:
      - source_cfg.variable_groups -> source_cfg.variables
      - source_cfg.temporal_postprocess.output_variable_groups
        -> source_cfg.temporal_postprocess.output_variables
    """
    for group_name, group_cfg in groups.items():
        if not isinstance(group_cfg, dict):
            raise TypeError(f"{target_name}_groups.{group_name} must be a dict.")

        template = group_cfg.get("template")
        if not isinstance(template, dict):
            raise TypeError(
                f"{target_name}_groups.{group_name}.template must be a dict."
            )

        items = _normalise_items(group_cfg)

        for item in items:
            context = dict(item)

            if "name" in template:
                item_name = _format_value(template["name"], context)
            elif "name" in context:
                item_name = str(context["name"])
            else:
                raise ValueError(
                    f"{target_name}_groups.{group_name} needs either "
                    "template.name or item.name."
                )

            item_cfg = _format_value(template, context)
            item_cfg.pop("name", None)

            item_cfg.setdefault("enabled", True)
            item_cfg.setdefault("generated", True)
            item_cfg.setdefault("generated_from_group", group_name)
            item_cfg.setdefault("generation_context", context)

            if item_name in target and not bool(
                group_cfg.get("overwrite_existing", False)
            ):
                raise ValueError(
                    f"{target_name} {item_name!r} already exists. "
                    f"Set overwrite_existing=true in group {group_name!r} "
                    "if this is intentional."
                )

            target[item_name] = item_cfg

    return target


def expand_variable_groups(source_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Expand source_cfg['variable_groups'] into source_cfg['variables'].

    This is provider-agnostic.
    """
    cfg = deepcopy(source_cfg)
    groups = cfg.get("variable_groups", {}) or {}

    if not groups:
        return cfg

    variables = cfg.setdefault("variables", {})

    _expand_groups_to_mapping(
        groups=groups,
        target=variables,
        target_name="variable",
    )

    return cfg


def expand_temporal_output_variable_groups(
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Expand:

      temporal_postprocess:
        output_variable_groups:
          ...

    into:

      temporal_postprocess:
        output_variables:
          ...
    """
    cfg = deepcopy(source_cfg)
    temporal_cfg = cfg.setdefault("temporal_postprocess", {})

    groups = temporal_cfg.get("output_variable_groups", {}) or {}

    if not groups:
        return cfg

    output_variables = temporal_cfg.setdefault("output_variables", {})

    _expand_groups_to_mapping(
        groups=groups,
        target=output_variables,
        target_name="temporal_output_variable",
    )

    return cfg


def expand_temporal_postprocess_variables(
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Add temporal_postprocess.output_variables to source_cfg['variables'].

    This means temporal aggregations become normal variables for clip/build.
    """
    cfg = deepcopy(source_cfg)

    temporal_cfg = cfg.get("temporal_postprocess", {}) or {}
    output_variables = temporal_cfg.get("output_variables", {}) or {}

    if not output_variables:
        return cfg

    variables = cfg.setdefault("variables", {})

    dataset_cfg = cfg.get("dataset", {}) or {}
    native_resolution_m = dataset_cfg.get("native_resolution_m")
    default_data_type = dataset_cfg.get("data_type")

    for variable_name, variable_cfg in output_variables.items():
        if not isinstance(variable_cfg, dict):
            raise TypeError(
                f"temporal_postprocess.output_variables.{variable_name} "
                "must be a dict."
            )

        if variable_name in variables:
            continue

        variables[variable_name] = {
            "enabled": True,
            "description": variable_cfg.get("description", variable_name),
            "unit": variable_cfg.get("unit"),
            "scale_factor": variable_cfg.get("scale_factor", 1.0),
            "valid_range": variable_cfg.get("valid_range"),
            "data_type": variable_cfg.get("data_type", default_data_type),
            "native_resolution_m": variable_cfg.get(
                "native_resolution_m",
                native_resolution_m,
            ),
            "source_filename": variable_cfg.get(
                "filename",
                f"{variable_name}.tif",
            ),
            "round_values": variable_cfg.get("round_values", False),
            "temporal": variable_cfg.get(
                "temporal",
                {"type": "temporal_aggregation"},
            ),

            # Important for temporal products:
            # some generated variables may legitimately not exist if there are no
            # selected dates for that metric/season. In that case clip/build should skip.
            "required": bool(variable_cfg.get("required", False)),

            "generated": True,
            "generated_from": "temporal_postprocess.output_variables",
        }

    return cfg


def expand_source_config(source_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Apply all provider-agnostic source config expansions.

    Recommended order:
      1. normal variable groups
      2. temporal output variable groups
      3. temporal output variables -> source variables
    """
    cfg = expand_variable_groups(source_cfg)
    cfg = expand_temporal_output_variable_groups(cfg)
    cfg = expand_temporal_postprocess_variables(cfg)
    return cfg
