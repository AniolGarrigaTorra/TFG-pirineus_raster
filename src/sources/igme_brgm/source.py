from __future__ import annotations

from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.igme_brgm.download import download_igme_brgm_raw_files
from src.sources.igme_brgm.clip import clip_igme_brgm_raw_files
from src.sources.igme_brgm.build import build_igme_brgm_features


def prepare_igme_brgm_raw_data(
    project_cfg: dict,
    source_cfg: dict,
) -> list[Path]:
    """
    Download and extract raw IGME/BRGM vector data.
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

    return download_igme_brgm_raw_files(
        source_cfg=source_cfg,
        raw_dir=raw_dir,
    )


def prepare_igme_brgm_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    """
    Clip IGME/BRGM vector layers to the configured AOI.
    """
    return clip_igme_brgm_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
    )


def prepare_igme_brgm_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Rasterize IGME/BRGM clipped vector layers to the target project grid.
    """
    return build_igme_brgm_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )