from __future__ import annotations

import re
from pathlib import Path


def safe_name(value: str) -> str:
    """
    Normalize strings for filenames and GeoPackage layer names.
    """
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def build_igme_brgm_zip_name(
    dataset_name: str,
    dataset_cfg: dict,
) -> str:
    """
    Return the local ZIP filename for one configured dataset.
    """
    if dataset_cfg.get("zip_filename"):
        return str(dataset_cfg["zip_filename"])

    return f"{safe_name(dataset_name)}.zip"


def build_igme_brgm_clipped_vector_name(
    dataset_name: str,
    layer_name: str,
    domain_name: str,
) -> str:
    """
    Return clipped vector filename.
    """
    return (
        f"igme_brgm_{safe_name(dataset_name)}_"
        f"{safe_name(layer_name)}_{safe_name(domain_name)}.gpkg"
    )


def build_igme_brgm_feature_name(
    provider: str,
    product: str,
    feature_name: str,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    """
    Return final raster feature filename.
    """
    return (
        f"{safe_name(provider)}_{safe_name(product)}_"
        f"{safe_name(feature_name)}_"
        f"{safe_name(domain_name)}_{int(target_resolution_m)}m.tif"
    )


def build_igme_brgm_legend_name(
    provider: str,
    product: str,
    feature_name: str,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    """
    Return legend CSV filename for a categorical raster.
    """
    return Path(
        f"{safe_name(provider)}_{safe_name(product)}_"
        f"{safe_name(feature_name)}_"
        f"{safe_name(domain_name)}_{int(target_resolution_m)}m_legend.csv"
    )