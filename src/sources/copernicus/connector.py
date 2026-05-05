from __future__ import annotations

from pathlib import Path

from src.sources.base import RasterSourceConnector
from src.sources.copernicus.source import (
    prepare_copernicus_raw_data,
    prepare_copernicus_clipped_data,
    prepare_copernicus_features,
)


class CopernicusConnector(RasterSourceConnector):
    """
    Connector for Copernicus raster products.

    Initial scope:
      - static_single
      - static_multi

    This is intentionally generic and not tied to one Copernicus product.
    Product-specific details should live in YAML configs, not in Python code.
    """

    provider = "copernicus"

    def download(
        self,
        project_cfg: dict,
        source_cfg: dict,
    ) -> list[Path]:
        return prepare_copernicus_raw_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
        )

    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_copernicus_clipped_data(
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
        return prepare_copernicus_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )