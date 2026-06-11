from __future__ import annotations

from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.generic_raster.build import build_generic_raster_features
from src.sources.generic_raster.clip import clip_generic_raster_raw_files
from src.sources.generic_raster.download import download_generic_raster_raw_files


def prepare_generic_raster_raw_data(
    project_cfg: dict,
    source_cfg: dict,
    provider: str | None = None,
    required_variables: set[str] | None = None,
) -> list[Path]:
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=str(processing["source_resolution"]),
    )
    return download_generic_raster_raw_files(
        source_cfg=source_cfg,
        raw_dir=raw_dir,
        provider=provider,
        required_variables=required_variables,
    )


def prepare_generic_raster_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    provider: str | None = None,
) -> list[Path]:
    return clip_generic_raster_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        provider=provider,
    )


def prepare_generic_raster_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
    provider: str | None = None,
) -> list[Path]:
    return build_generic_raster_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
        provider=provider,
    )
