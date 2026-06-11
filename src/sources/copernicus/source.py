from __future__ import annotations

from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.copernicus.download import download_copernicus_raw_files
from src.sources.copernicus.clip import clip_copernicus_raw_files
from src.sources.copernicus.build import build_copernicus_features


def prepare_copernicus_raw_data(
    project_cfg: dict,
    source_cfg: dict,
    required_variables: set[str] | None = None,
) -> list[Path]:
    """
    Prepare raw Copernicus source files.

    This stage downloads files or validates that manually downloaded files
    already exist.
    
    Parameters
    ----------
    required_variables : set[str] | None
        If provided, only download these variables.
    """
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    provider = source["provider"]
    product = source["product"]
    source_resolution = str(processing["source_resolution"])

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        source_resolution=source_resolution,
    )

    return download_copernicus_raw_files(
        source_cfg=source_cfg,
        raw_dir=raw_dir,
        required_variables=required_variables,
    )


def prepare_copernicus_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    """
    Clip raw Copernicus rasters to the configured clipping AOI.
    """
    return clip_copernicus_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
    )


def prepare_copernicus_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned Copernicus features.
    """
    return build_copernicus_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )