from __future__ import annotations

from pathlib import Path

from src.io.config import load_yaml, resolve_path
from src.sources.registry import get_source_connector
from src.pipeline.source_overrides import apply_run_overrides_to_source_cfg
from src.pipeline.variable_expansion import expand_source_config
from src.workbench.compiler import compile_source_config_for_run


VALID_STAGES = {"download", "clip", "build", "all"}


def _print_header(
    source_cfg: dict,
    stage: str,
    project_config_path: str | Path,
    source_config_path: str | Path,
) -> None:
    source = source_cfg["source"]

    print("==============================")
    print("Pirineus Raster Source Pipeline")
    print(f"Provider: {source['provider']}")
    print(f"Product:  {source['product']}")
    print(f"Source:   {source.get('id', 'unknown')}")
    print(f"Stage:    {stage}")
    print("------------------------------")
    print(f"Project config: {project_config_path}")
    print(f"Source config:  {source_config_path}")
    print("==============================")


def _print_paths(
    title: str,
    paths: list[Path],
    max_items: int = 10,
) -> None:
    print("==============================")
    print(title)
    print(f"Total files: {len(paths)}")

    for path in paths[:max_items]:
        print(f"  - {path}")

    if len(paths) > max_items:
        print(f"  ... and {len(paths) - max_items} more")

    print("==============================")


def _load_domain_configs(source_cfg: dict) -> tuple[dict, dict]:
    """
    Load clipping AOI and output AOI configs declared by the source config.

    Source configs currently define their own domains:

      domains:
        clip_aoi_config: ...
        output_aoi_config: ...

    The run_config orchestrates which sources are executed, while each source
    still defines the clipping/output domain it needs.
    """
    domains_cfg = source_cfg["domains"]
    source_config_path = source_cfg.get("_config_path")

    clip_aoi_path = resolve_path(
        domains_cfg["clip_aoi_config"],
        base_path=source_config_path,
        must_exist=True,
    )
    output_aoi_path = resolve_path(
        domains_cfg["output_aoi_config"],
        base_path=source_config_path,
        must_exist=True,
    )

    clip_aoi_cfg = load_yaml(clip_aoi_path)
    output_aoi_cfg = load_yaml(output_aoi_path)

    return clip_aoi_cfg, output_aoi_cfg


def _normalize_single_stage(stage: str) -> list[str]:
    """
    Normalize one source-level stage.

    At source-level, 'all' expands to:
      download -> clip -> build
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage '{stage}'. Valid stages are: {sorted(VALID_STAGES)}"
        )

    if stage == "all":
        return ["download", "clip", "build"]

    return [stage]


def run_source_pipeline(
    project_config_path: str,
    source_config_path: str,
    stage: str,
    run_cfg: dict | None = None,
    source_entry: dict | None = None,
) -> list:
    """
    Run one source pipeline stage.

    This is the source-level runner used by:
      - pirineus-raster run-source
      - dataset-level runs from src.pipeline.dataset

    Parameters
    ----------
    project_config_path:
        Path to the global project configuration YAML.

    source_config_path:
        Path to one source configuration YAML.

    stage:
        One of: download, clip, build, all.

    Returns
    -------
    list[Path]
        Paths generated or prepared by the final executed stage.
    """
    project_config_path = resolve_path(project_config_path, must_exist=True)
    source_config_path = resolve_path(source_config_path, must_exist=True)

    project_cfg = load_yaml(project_config_path)
    project_cfg["_config_path"] = str(project_config_path)
    source_cfg = load_yaml(source_config_path)
    source_cfg["_config_path"] = str(source_config_path)

    source_cfg = apply_run_overrides_to_source_cfg(
        source_cfg=source_cfg,
        run_cfg=run_cfg,
    )
    source_cfg = expand_source_config(source_cfg)
    source_cfg = compile_source_config_for_run(
        source_cfg=source_cfg,
        source_entry=source_entry,
    )
    source_cfg = expand_source_config(source_cfg)

    source = source_cfg["source"]
    provider = source["provider"]

    _print_header(
        source_cfg=source_cfg,
        stage=stage,
        project_config_path=project_config_path,
        source_config_path=source_config_path,
    )

    connector = get_source_connector(provider)

    final_paths: list[Path] = []

    for current_stage in _normalize_single_stage(stage):
        if current_stage == "download":
            raw_paths = connector.download(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
            )
            _print_paths("Raw files ready", raw_paths)
            final_paths = raw_paths

        elif current_stage == "clip":
            clip_aoi_cfg, _ = _load_domain_configs(source_cfg)

            clipped_paths = connector.clip(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                clip_aoi_cfg=clip_aoi_cfg,
            )
            _print_paths("Clipped files ready", clipped_paths)
            final_paths = clipped_paths

        elif current_stage == "build":
            clip_aoi_cfg, output_aoi_cfg = _load_domain_configs(source_cfg)

            feature_paths = connector.build(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                clip_aoi_cfg=clip_aoi_cfg,
                output_aoi_cfg=output_aoi_cfg,
            )
            _print_paths("Feature files ready", feature_paths)
            final_paths = feature_paths

        else:
            raise ValueError(f"Unexpected normalized stage: {current_stage}")

    print("Pirineus Raster source pipeline finished successfully.")

    return final_paths
