from __future__ import annotations

from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.openstreetmap.build import build_osm_features
from src.sources.openstreetmap.clip import clip_osm_raw_files
from src.sources.openstreetmap.download import download_osm_raw_files
from src.sources.openstreetmap.naming import get_source_resolution


def prepare_osm_raw_data(project_cfg: dict, source_cfg: dict, required_variables: set[str] | None = None) -> list[Path]:
    source = source_cfg["source"]
    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=get_source_resolution(source_cfg),
    )
    return download_osm_raw_files(source_cfg=source_cfg, raw_dir=raw_dir, required_variables=required_variables)


def prepare_osm_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    return clip_osm_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
    )


def prepare_osm_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    return build_osm_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )
