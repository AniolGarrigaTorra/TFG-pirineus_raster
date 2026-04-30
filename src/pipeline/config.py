from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io.config import load_yaml


VALID_STAGES = {"download", "clip", "build", "all"}


# =============================================================================
# Run config loading and validation
# =============================================================================


def load_run_config(run_config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a run configuration YAML.

    A run config describes one complete dataset generation recipe:
      - global project config
      - AOI
      - target resolution
      - source configs to execute
      - optional derived features
      - dataset output configuration
    """
    cfg = load_yaml(run_config_path)
    validate_run_config(cfg, run_config_path)
    return cfg


def validate_run_config(
    cfg: dict[str, Any],
    run_config_path: str | Path | None = None,
) -> None:
    location = f" in {run_config_path}" if run_config_path is not None else ""

    if "run" not in cfg:
        raise ValueError(f"Missing required top-level key 'run'{location}.")

    if "sources" not in cfg:
        raise ValueError(f"Missing required top-level key 'sources'{location}.")

    run_cfg = cfg["run"]

    required_run_keys = ["name", "project_config"]
    for key in required_run_keys:
        if key not in run_cfg:
            raise ValueError(f"Missing required key 'run.{key}'{location}.")

    validate_stages(
        run_cfg.get("stages", ["build"]),
        context="run.stages",
    )

    sources = cfg["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"'sources' must be a non-empty list{location}.")

    for idx, source_entry in enumerate(sources):
        if not isinstance(source_entry, dict):
            raise ValueError(
                f"Each source entry must be a dictionary. "
                f"Invalid entry at sources[{idx}]{location}."
            )

        if "config" not in source_entry:
            raise ValueError(
                f"Missing required key 'config' in sources[{idx}]{location}."
            )

        stages = source_entry.get("stages", run_cfg.get("stages", ["build"]))
        validate_stages(
            stages,
            context=f"sources[{idx}].stages",
        )

    validate_outputs_config(
        cfg.get("outputs", {}),
        location=location,
    )

    validate_derived_features_config(
        cfg.get("derived_features", []),
        location=location,
    )


def validate_outputs_config(
    outputs_cfg: Any,
    location: str = "",
) -> None:
    if outputs_cfg is None:
        return

    if not isinstance(outputs_cfg, dict):
        raise ValueError(f"'outputs' must be a dictionary{location}.")

    boolean_keys = [
        "copy_rasters",
        "overwrite_existing",
        "write_run_summary",
        "write_manifest",
    ]

    for key in boolean_keys:
        if key in outputs_cfg and not isinstance(outputs_cfg[key], bool):
            raise ValueError(f"'outputs.{key}' must be true or false{location}.")


def validate_derived_features_config(
    derived_features: Any,
    location: str = "",
) -> None:
    if derived_features in [None, []]:
        return

    if not isinstance(derived_features, list):
        raise ValueError(f"'derived_features' must be a list{location}.")

    for idx, item in enumerate(derived_features):
        if not isinstance(item, dict):
            raise ValueError(
                f"derived_features[{idx}] must be a dictionary{location}."
            )

        for key in ["name", "expression", "inputs"]:
            if key not in item:
                raise ValueError(
                    f"Missing required key 'derived_features[{idx}].{key}'{location}."
                )

        if not isinstance(item["inputs"], dict) or not item["inputs"]:
            raise ValueError(
                f"derived_features[{idx}].inputs must be a non-empty dictionary{location}."
            )


def validate_stages(
    stages: Any,
    context: str = "stages",
) -> None:
    if isinstance(stages, str):
        stages = [stages]

    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{context} must be a non-empty string or list.")

    invalid = [stage for stage in stages if stage not in VALID_STAGES]
    if invalid:
        raise ValueError(
            f"Invalid stage(s) in {context}: {invalid}. "
            f"Valid stages are: {sorted(VALID_STAGES)}"
        )


def normalize_stages(
    stages: str | list[str] | None,
) -> list[str]:
    """
    Normalize stages to a list.

    'all' expands to:
      ['download', 'clip', 'build']
    """
    if stages is None:
        stages = ["build"]

    if isinstance(stages, str):
        stages = [stages]

    normalized: list[str] = []

    for stage in stages:
        if stage == "all":
            normalized.extend(["download", "clip", "build"])
        else:
            normalized.append(stage)

    # Preserve order, remove duplicates.
    result: list[str] = []
    for stage in normalized:
        if stage not in result:
            result.append(stage)

    return result


# =============================================================================
# Run config accessors
# =============================================================================


def get_run_name(cfg: dict[str, Any]) -> str:
    return str(cfg["run"]["name"])


def get_project_config_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["run"].get("project_config", "configs/project.yaml"))


def get_run_aoi_config_path(cfg: dict[str, Any]) -> Path | None:
    value = cfg["run"].get("aoi_config")
    if value is None:
        return None
    return Path(value)


def get_run_resolution_m(cfg: dict[str, Any]) -> int | None:
    value = cfg["run"].get("resolution_m")
    if value is None:
        return None
    return int(value)


def get_dataset_dir(cfg: dict[str, Any]) -> Path:
    outputs_cfg = cfg.get("outputs", {})
    run_name = get_run_name(cfg)

    return Path(
        outputs_cfg.get(
            "dataset_dir",
            f"data_processed/datasets/{run_name}",
        )
    )


def get_source_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cfg["sources"])


# =============================================================================
# Source config helpers
# =============================================================================


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