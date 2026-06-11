from __future__ import annotations

from pathlib import Path

from src.sources.base import RasterSourceConnector
from src.sources.worldclim.source import (
    prepare_worldclim_raw_data,
    prepare_worldclim_clipped_data,
    prepare_worldclim_features,
)


class WorldClimConnector(RasterSourceConnector):
    """
    Connector for WorldClim raster products.

    This class keeps the existing working WorldClim implementation,
    but exposes it through the generic connector interface.
    """

    provider = "worldclim"

    def download(
        self,
        project_cfg: dict,
        source_cfg: dict,
        required_variables: set[str] | None = None,
    ) -> list[Path]:
        return prepare_worldclim_raw_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            required_variables=required_variables,
        )

    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_worldclim_clipped_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
        )

    def build(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
        output_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_worldclim_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )