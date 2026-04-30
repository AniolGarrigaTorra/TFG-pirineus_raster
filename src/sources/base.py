from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class RasterSourceConnector(ABC):
    """
    Base interface for all raster source connectors.

    A connector knows how to prepare one external data source:
    - download raw data
    - clip raw data to an intermediate AOI
    - build final grid-aligned features

    The central pipeline should not know provider-specific details.
    """

    provider: str

    @abstractmethod
    def download(
        self,
        project_cfg: dict,
        source_cfg: dict,
    ) -> list[Path]:
        """
        Download or validate raw source files.
        """
        raise NotImplementedError

    @abstractmethod
    def clip(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
    ) -> list[Path]:
        """
        Clip raw source files to the configured clipping AOI.
        """
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
        output_aoi_cfg: dict,
    ) -> list[Path]:
        """
        Build final processed features aligned to the target grid.
        """
        raise NotImplementedError