from __future__ import annotations

from pathlib import Path

from src.sources.base import RasterSourceConnector
from src.sources.pdca.source import (
    prepare_pdca_clipped_data,
    prepare_pdca_features,
    prepare_pdca_raw_data,
)


class PdcaConnector(RasterSourceConnector):
    """Connector for the Pyrenean Digital Climate Atlas dataset."""

    provider = "pdca"

    def download(self, project_cfg: dict, source_cfg: dict) -> list[Path]:
        return prepare_pdca_raw_data(project_cfg=project_cfg, source_cfg=source_cfg)

    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        return prepare_pdca_clipped_data(
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
        return prepare_pdca_features(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_cfg=clip_aoi_cfg,
            output_aoi_cfg=output_aoi_cfg,
        )
