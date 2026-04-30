from __future__ import annotations


def get_enabled_variables(source_cfg: dict) -> list[str]:
    """
    Return enabled variable names from source_cfg['variables'].
    """
    variables_cfg = source_cfg.get("variables", {})

    enabled = [
        variable
        for variable, cfg in variables_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled variables found in source config.")

    return enabled


def get_enabled_variable_items(source_cfg: dict) -> list[tuple[str, dict]]:
    """
    Return enabled variable config items from source_cfg['variables'].
    """
    variables_cfg = source_cfg.get("variables", {})

    enabled = [
        (variable, cfg)
        for variable, cfg in variables_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled variables found in source config.")

    return enabled


def get_enabled_index_items(source_cfg: dict) -> list[tuple[str, dict]]:
    """
    Return enabled index config items from source_cfg['indices'].
    """
    indices_cfg = source_cfg.get("indices", {})

    enabled = [
        (index_name, cfg)
        for index_name, cfg in indices_cfg.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled indices found in source config.")

    return enabled


def get_static_layer_items(source_cfg: dict) -> list[tuple[str, dict]]:
    """
    Return enabled static layer items.

    Supported structures:
      - static_single:
          uses source_cfg['variables']
          example: elev

      - static_index_set:
          uses source_cfg['indices']
          example: bio1...bio19
    """
    layer_structure = source_cfg.get("dataset", {}).get("layer_structure")

    if layer_structure == "static_single":
        return get_enabled_variable_items(source_cfg)

    if layer_structure == "static_index_set":
        return get_enabled_index_items(source_cfg)

    raise NotImplementedError(
        "get_static_layer_items only supports static_single or static_index_set. "
        f"Got layer_structure={layer_structure}"
    )


def get_temporal_aggregations(source_cfg: dict) -> list[dict]:
    aggregations = source_cfg.get("temporal_aggregations", [])

    if not aggregations:
        raise ValueError("No temporal_aggregations found in source config.")

    return aggregations


def aggregation_applies_to_variable(
    aggregation_cfg: dict,
    variable: str,
) -> bool:
    variables = aggregation_cfg.get("variables")

    if variables is None:
        return True

    return variable in variables


def years_from_range(year_range: list[int]) -> list[int]:
    if len(year_range) != 2:
        raise ValueError(f"Year range must have two values: {year_range}")

    start_year, end_year = int(year_range[0]), int(year_range[1])

    if start_year > end_year:
        raise ValueError(f"Invalid year range: {year_range}")

    return list(range(start_year, end_year + 1))


def get_time_series_metric_name(aggregation_cfg: dict) -> str:
    if "output_metric_name" in aggregation_cfg:
        return aggregation_cfg["output_metric_name"]

    if "metric" in aggregation_cfg:
        return aggregation_cfg["metric"]

    within = aggregation_cfg.get("within_year_metric")
    across = aggregation_cfg.get("across_year_metric")

    if within and across:
        return f"{across}_annual_{within}"

    raise ValueError(f"Cannot infer metric name from aggregation: {aggregation_cfg}")