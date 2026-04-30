from __future__ import annotations

from pathlib import Path

from src.io.config import load_yaml
from src.sources.registry import get_source_connector


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


def _print_paths(title: str, paths: list[Path], max_items: int = 10) -> None:
    print("==============================")
    print(title)
    print(f"Total files: {len(paths)}")

    for path in paths[:max_items]:
        print(f"  - {path}")

    if len(paths) > max_items:
        print(f"  ... and {len(paths) - max_items} more")

    print("==============================")


def _load_domain_configs(source_cfg: dict) -> tuple[dict, dict]:
    domains_cfg = source_cfg["domains"]

    clip_aoi_cfg = load_yaml(domains_cfg["clip_aoi_config"])
    output_aoi_cfg = load_yaml(domains_cfg["output_aoi_config"])

    return clip_aoi_cfg, output_aoi_cfg


def run_source_pipeline(
    project_config_path: str | Path = "configs/project.yaml",
    source_config_path: str | Path = "configs/sources/worldclim/worldclim_v2_1_climate_normals.yaml",
    stage: str = "build",
) -> list[Path]:
    """
    Run one source pipeline stage.

    This is still the source-level runner.
    The dataset-level runner lives in src.pipeline.run_orchestrator.
    """
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage '{stage}'. Valid stages are: {sorted(VALID_STAGES)}"
        )

    project_cfg = load_yaml(project_config_path)
    source_cfg = load_yaml(source_config_path)

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

    if stage in {"download", "all"}:
        raw_paths = connector.download(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
        )
        _print_paths("Raw files ready", raw_paths)
        final_paths = raw_paths

    if stage in {"clip", "all"}:
        clip_aoi_cfg, _ = _load_domain_configs(source_cfg)

        clipped_paths = connector.clip(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
        )
        _print_paths("Clipped files ready", clipped_paths)
        final_paths = clipped_paths

    if stage in {"build", "all"}:
        clip_aoi_cfg, output_aoi_cfg = _load_domain_configs(source_cfg)

        feature_paths = connector.build(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )
        _print_paths("Feature files ready", feature_paths)
        final_paths = feature_paths

    print("Pirineus Raster source pipeline finished successfully.")

    return final_paths