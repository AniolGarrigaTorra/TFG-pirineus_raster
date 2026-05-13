from __future__ import annotations

from pathlib import Path

from src.sources.base import RasterSourceConnector
from src.sources.igme_brgm.source import (
    prepare_igme_brgm_raw_data,
    prepare_igme_brgm_clipped_data,
    prepare_igme_brgm_features,
)


class IgmeBrgmConnector(RasterSourceConnector):
    """
    Connector for IGME/BRGM geological vector products.

    This provider starts with the transboundary Pyrenees geological maps
    at 1:400,000 scale and is designed to support future regional geology
    products with the same source interface.
    """

    provider = "igme_brgm"

    def download(
        self,
        project_cfg: dict,
        source_cfg: dict,
    ) -> list[Path]:
        return prepare_igme_brgm_raw_data(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
        )

    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_igme_brgm_clipped_data(
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
        return prepare_igme_brgm_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )