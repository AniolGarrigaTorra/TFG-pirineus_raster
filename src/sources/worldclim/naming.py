from pathlib import Path


SUPPORTED_WORLDCLIM_RESOLUTIONS = {"10m", "5m", "2.5m", "30s"}

SUPPORTED_WORLDCLIM_VARIABLES = {
    "tmin",
    "tmax",
    "tavg",
    "prec",
    "srad",
    "wind",
    "vapr",
}


def validate_worldclim_resolution(source_resolution: str) -> None:
    if source_resolution not in SUPPORTED_WORLDCLIM_RESOLUTIONS:
        raise ValueError(
            f"Unsupported WorldClim resolution: {source_resolution}. "
            f"Supported: {sorted(SUPPORTED_WORLDCLIM_RESOLUTIONS)}"
        )


def validate_worldclim_variable(variable: str) -> None:
    if variable not in SUPPORTED_WORLDCLIM_VARIABLES:
        raise ValueError(
            f"Unsupported WorldClim variable: {variable}. "
            f"Supported: {sorted(SUPPORTED_WORLDCLIM_VARIABLES)}"
        )


def build_worldclim_zip_name(source_resolution: str, variable: str) -> str:
    """
    Example:
    wc2.1_10m_tmin.zip
    wc2.1_30s_prec.zip
    """
    validate_worldclim_resolution(source_resolution)
    validate_worldclim_variable(variable)

    return f"wc2.1_{source_resolution}_{variable}.zip"


def build_worldclim_download_url(
    base_url: str,
    source_resolution: str,
    variable: str,
) -> str:
    zip_name = build_worldclim_zip_name(source_resolution, variable)
    return f"{base_url.rstrip('/')}/{zip_name}"


def build_worldclim_zip_path(
    raw_dir: Path,
    source_resolution: str,
    variable: str,
) -> Path:
    zip_name = build_worldclim_zip_name(source_resolution, variable)
    return raw_dir / zip_name


def build_worldclim_clipped_month_name(
    source_resolution: str,
    variable: str,
    month: int,
    domain_name: str,
) -> str:
    """
    Example:
    wc2.1_10m_tmin_01_experimental_pallars_sobira.tif
    """
    validate_worldclim_resolution(source_resolution)
    validate_worldclim_variable(variable)

    return f"wc2.1_{source_resolution}_{variable}_{month:02d}_{domain_name}.tif"


def build_worldclim_feature_name(
    provider: str,
    product: str,
    variable: str,
    metric: str,
    start_month: int,
    end_month: int,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    """
    Example:
    worldclim_v2_1_base_tmin_mean_m01-m12_experimental_pallars_sobira_100m.tif
    """
    return (
        f"{provider}_{product}_{variable}_{metric}_"
        f"m{start_month:02d}-m{end_month:02d}_"
        f"{domain_name}_{int(target_resolution_m)}m.tif"
    )