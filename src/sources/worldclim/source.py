from pathlib import Path

from src.io.paths import get_source_raw_dir
from src.sources.worldclim.download import download_worldclim_raw_files
from src.sources.worldclim.clip import clip_worldclim_raw_files
from src.pipeline.raster_pipeline import build_worldclim_features


def prepare_worldclim_raw_data(
    project_cfg: dict,
    source_cfg: dict,
) -> list[Path]:
    """
    Prepare raw global WorldClim ZIP files.

    This downloads or validates raw ZIPs.
    """
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        source_resolution=source_resolution,
    )

    return download_worldclim_raw_files(
        source_cfg=source_cfg,
        raw_dir=raw_dir,
    )


def prepare_worldclim_clipped_data(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    """
    Prepare clipped WorldClim monthly rasters for the configured clipping AOI.
    """
    return clip_worldclim_raw_files(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
    )


def prepare_worldclim_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned WorldClim features.
    """
    return build_worldclim_features(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )