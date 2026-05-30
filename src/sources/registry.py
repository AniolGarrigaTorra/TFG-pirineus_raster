from __future__ import annotations

from src.sources.base import RasterSourceConnector
from src.sources.worldclim.connector import WorldClimConnector
from src.sources.copernicus.connector import CopernicusConnector
from src.sources.generic_raster.connector import GenericRasterConnector
from src.sources.igme_brgm.connector import IgmeBrgmConnector
from src.sources.openstreetmap.connector import OpenStreetMapConnector
from src.sources.pdca.connector import PdcaConnector


_SOURCE_CONNECTORS: dict[str, RasterSourceConnector] = {
    "worldclim": WorldClimConnector(),
    "copernicus": CopernicusConnector(),
    "esa_cci": GenericRasterConnector(provider="esa_cci"),
    "ghsl": GenericRasterConnector(provider="ghsl"),
    "igme_brgm": IgmeBrgmConnector(),
    "openstreetmap": OpenStreetMapConnector(),
    "pdca": PdcaConnector(),
}


def get_source_connector(provider: str) -> RasterSourceConnector:
    """
    Return the connector associated with a provider name.
    """
    try:
        return _SOURCE_CONNECTORS[provider]
    except KeyError as exc:
        available = ", ".join(sorted(_SOURCE_CONNECTORS))
        raise NotImplementedError(
            f"No raster source connector implemented for provider='{provider}'. "
            f"Available providers: {available}"
        ) from exc


def list_source_connectors() -> list[str]:
    """
    Return the list of available provider names.
    """
    return sorted(_SOURCE_CONNECTORS)
