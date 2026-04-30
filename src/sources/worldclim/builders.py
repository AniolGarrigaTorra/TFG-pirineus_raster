from __future__ import annotations

from pathlib import Path

from src.sources.worldclim.future_monthly_builder import (
    build_worldclim_future_monthly_multiband_features,
)
from src.sources.worldclim.monthly_climatology_builder import (
    build_worldclim_monthly_features,
)
from src.sources.worldclim.monthly_time_series_builder import (
    build_worldclim_monthly_time_series_features,
)
from src.sources.worldclim.static_builder import build_worldclim_static_features


def build_worldclim_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Route WorldClim source configs to the correct provider-specific builder.

    This file intentionally stays small.
    Shared raster processing belongs in src.pipeline.
    WorldClim-specific filename/location logic belongs in the builder modules.
    """
    layer_structure = source_cfg["dataset"]["layer_structure"]

    if layer_structure in {"static_index_set", "static_single"}:
        return build_worldclim_static_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "monthly_climatology":
        return build_worldclim_monthly_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "monthly_time_series":
        return build_worldclim_monthly_time_series_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    if layer_structure == "future_monthly_multiband":
        return build_worldclim_future_monthly_multiband_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )

    raise NotImplementedError(
        f"Unsupported WorldClim layer_structure: {layer_structure}"
    )