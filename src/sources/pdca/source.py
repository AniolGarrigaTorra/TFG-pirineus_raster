from __future__ import annotations

from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.pdca.build import build_pdca_features
from src.sources.pdca.clip import clip_pdca_raw_files
from src.sources.pdca.download import download_pdca_raw_files


def prepare_pdca_raw_data(project_cfg: dict, source_cfg: dict, required_variables: set[str] | None = None) -> list[Path]:
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=processing["source_resolution"],
    )

    return download_pdca_raw_files(source_cfg=source_cfg, raw_dir=raw_dir, required_variables=required_variables)


def prepare_pdca_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    return clip_pdca_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
    )


def prepare_pdca_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    return build_pdca_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )
