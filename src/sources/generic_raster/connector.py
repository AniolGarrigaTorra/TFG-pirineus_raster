from __future__ import annotations

from pathlib import Path

from src.sources.base import RasterSourceConnector
from src.sources.generic_raster.source import (
    prepare_generic_raster_clipped_data,
    prepare_generic_raster_features,
    prepare_generic_raster_raw_data,
)


class GenericRasterConnector(RasterSourceConnector):
    """
    Connector for provider-specific static rasters whose details live in YAML.
    """

    def __init__(self, provider: str):
        self.provider = provider

    def download(self, project_cfg: dict, source_cfg: dict, required_variables: set[str] | None = None) -> list[Path]:
        return prepare_generic_raster_raw_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            provider=self.provider,
            required_variables=required_variables,
        )

    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_generic_raster_clipped_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            provider=self.provider,
        )

    def build(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
        output_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_generic_raster_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
            provider=self.provider,
        )
