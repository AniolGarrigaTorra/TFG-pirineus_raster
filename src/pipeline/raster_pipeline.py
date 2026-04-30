from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RasterFeatureBuilder(Protocol):
    """
    Protocol for provider-specific raster feature builders.

    Provider-specific modules, such as src.sources.worldclim.builders,
    should expose functions/classes that satisfy this interface.
    """

    def __call__(
        self,
        project_cfg: dict,
        source_cfg: dict,
        clip_aoi_cfg: dict,
        output_aoi_cfg: dict,
    ) -> list[Path]:
        ...


def build_features_with_builder(
    builder: RasterFeatureBuilder,
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Execute a provider-specific feature builder through a generic interface.

    This file intentionally contains no provider-specific imports.
    """
    return builder(
        project_cfg=project_cfg,
        source_cfg=source_cfg,
        clip_aoi_cfg=clip_aoi_cfg,
        output_aoi_cfg=output_aoi_cfg,
    )