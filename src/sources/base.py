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
        required_variables: set[str] | None = None,
    ) -> list[Path]:
        """
        Download or validate raw source files.
        
        Parameters
        ----------
        project_cfg : dict
            Project configuration
        source_cfg : dict
            Source configuration
        required_variables : set[str] | None
            If provided, only download these variables (filter by variable name).
            If None or empty, download all enabled variables (backward compatible).
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