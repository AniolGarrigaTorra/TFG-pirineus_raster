from __future__ import annotations

import shutil
from pathlib import Path

from src.io.config import load_yaml, resolve_path
from src.io.paths import get_source_raw_dir
from src.pipeline.project_overrides import apply_run_overrides_to_project_cfg
from src.pipeline.progress import progress_log
from src.pipeline.source_overrides import (
    apply_run_overrides_to_source_cfg,
    normalize_source_domains,
)
from src.pipeline.variable_expansion import expand_source_config
from src.sources.registry import get_source_connector
from src.workbench.compiler import compile_source_config_for_run


VALID_STAGES = {"download", "clip", "build", "all"}


def _print_header(
    source_cfg: dict,
    stage: str,
    project_config_path: str | Path,
    source_config_path: str | Path,
) -> None:
    source = source_cfg["source"]
    progress_log(
        "Source pipeline | "
        f"provider={source['provider']} | product={source['product']} | "
        f"source={source.get('id', 'unknown')} | stage={stage}"
    )
    progress_log(f"Project config: {project_config_path}")
    progress_log(f"Source config: {source_config_path}")


def _print_paths(
    title: str,
    paths: list[Path],
    max_items: int = 10,
) -> None:
    progress_log(f"{title}: {len(paths)} files")

    for path in paths[:max_items]:
        progress_log(f"  - {path}")

    if len(paths) > max_items:
        progress_log(f"  ... and {len(paths) - max_items} more")


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


def _cleanup_raw_after_clip(
    project_cfg: dict,
    source_cfg: dict,
) -> None:
    download_cfg = source_cfg.get("download", {}) or {}
    if not bool(download_cfg.get("delete_raw_after_clip", False)):
        return

    source = source_cfg["source"]
    processing = source_cfg.get("processing", {}) or {}
    source_resolution = processing.get("source_resolution")
    if source_resolution is None:
        progress_log("[clip] Raw cleanup skipped: source_resolution is not defined.")
        return

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=str(source_resolution),
    )

    if not raw_dir.exists():
        progress_log(f"[clip] Raw cleanup skipped: {raw_dir} does not exist.")
        return

    progress_log(f"[clip] Removing raw source directory after successful clip: {raw_dir}")
    shutil.rmtree(raw_dir)


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
    project_cfg = apply_run_overrides_to_project_cfg(project_cfg, run_cfg)
    source_cfg = load_yaml(source_config_path)
    source_cfg["_config_path"] = str(source_config_path)
    source_cfg = normalize_source_domains(source_cfg)

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
            _cleanup_raw_after_clip(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
            )
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

    progress_log("Pirineus Raster source pipeline finished successfully.")

    return final_paths
