from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io.config import load_yaml


VALID_STAGES = {"download", "clip", "build", "all"}


def load_run_config(run_config_path: str | Path) -> dict[str, Any]:
    """
    Load and validate a run configuration YAML.
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

    validate_stages(run_cfg.get("stages", ["build"]), context="run.stages")

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
        validate_stages(stages, context=f"sources[{idx}].stages")

    validate_outputs_config(cfg.get("outputs", {}), location=location)


def validate_outputs_config(outputs_cfg: Any, location: str = "") -> None:
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


def validate_stages(stages: Any, context: str = "stages") -> None:
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


def normalize_stages(stages: str | list[str] | None) -> list[str]:
    """
    Normalize stages to a list.

    'all' expands to ['download', 'clip', 'build'].
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

    result: list[str] = []
    for stage in normalized:
        if stage not in result:
            result.append(stage)

    return result


def get_run_name(cfg: dict[str, Any]) -> str:
    return str(cfg["run"]["name"])


def get_project_config_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["run"].get("project_config", "configs/project.yaml"))


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