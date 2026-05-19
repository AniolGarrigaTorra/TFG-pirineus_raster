from __future__ import annotations

import re
from pathlib import Path
from typing import Any


RASTER_SUFFIXES = {".tif", ".tiff", ".geotiff", ".gtif"}

# PDCA temporal layers observed in Zenodo record 1186639.
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
SEASONS = ["winter", "spring", "summer", "autumn"]
ANNUAL = "annual"
TEMPORAL_LAYERS = [ANNUAL, *MONTHS, *SEASONS]


PDCA_VARIABLES: dict[str, dict[str, Any]] = {
    "Temperature_min": {
        "variable_key": "tmin",
        "canonical_prefix": "temperature_min",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "Temperature_max": {
        "variable_key": "tmax",
        "canonical_prefix": "temperature_max",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "Temperature_mean": {
        "variable_key": "tmean",
        "canonical_prefix": "temperature_mean",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "Precipitation": {
        "variable_key": "prec",
        "canonical_prefix": "precipitation",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "PET": {
        "variable_key": "pet",
        "canonical_prefix": "pet",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "Water_availability": {
        "variable_key": "water_availability",
        "canonical_prefix": "water_availability",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "Pot_solar_rad": {
        "variable_key": "psr",
        "canonical_prefix": "pot_solar_rad",
        "temporal_layers": TEMPORAL_LAYERS,
    },
    "GDD": {
        "variable_key": "gdd",
        "canonical_prefix": "gdd",
        "temporal_layers": [None],
    },
}

def raw_zip_path(raw_dir: Path, filename: str) -> Path:
    """
    Return the expected local path for a raw PDCA ZIP downloaded from Zenodo.

    Kept here because download.py uses this helper to decide where each
    Zenodo file should be stored.
    """
    return raw_dir / filename

def safe_slug(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "layer"


def strip_pyrenees_suffix(value: str) -> str:
    """Remove dataset-area suffixes already represented by the AOI/domain name."""
    value = safe_slug(value)
    value = re.sub(r"_pyrenees$", "", value)
    return value


def temporal_kind(period: str | None) -> str:
    if period is None:
        return "annual_index"
    p = safe_slug(period)
    if p == "annual":
        return "annual"
    if p in MONTHS:
        return "monthly"
    if p in SEASONS:
        return "seasonal"
    return "unknown"


def canonical_layer_id(canonical_prefix: str, period: str | None) -> str:
    if period is None:
        return safe_slug(canonical_prefix)
    return safe_slug(f"{canonical_prefix}_{period}")


def clipped_raster_name(layer_id: str, domain_name: str) -> str:
    return f"pdca_{layer_id}_{domain_name}.tif"


def feature_raster_name(
    provider: str,
    product: str,
    layer_id: str,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    return (
        f"{provider}_{product}_{layer_id}_"
        f"{domain_name}_{int(target_resolution_m)}m.tif"
    )


def variable_key_from_layer_id(layer_id: str, layer_metadata: dict[str, Any] | None = None) -> str:
    if layer_metadata and layer_metadata.get("variable_key"):
        return str(layer_metadata["variable_key"])

    layer = safe_slug(layer_id)
    for spec in PDCA_VARIABLES.values():
        prefix = spec["canonical_prefix"]
        if layer == prefix or layer.startswith(prefix + "_"):
            return str(spec["variable_key"])
    return "unknown"


def find_single_raster(folder: Path) -> Path:
    rasters = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in RASTER_SUFFIXES
        and not path.name.startswith("._")
    )
    if not rasters:
        raise FileNotFoundError(f"No raster found under expected PDCA folder: {folder}")
    if len(rasters) > 1:
        raise RuntimeError(
            f"Expected one raster under {folder}, found {len(rasters)}: "
            + ", ".join(str(p) for p in rasters[:10])
        )
    return rasters[0]
