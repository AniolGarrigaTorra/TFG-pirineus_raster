from pathlib import Path


SUPPORTED_WORLDCLIM_RESOLUTIONS = {"10m", "5m", "2.5m", "30s"}

MONTHLY_VARIABLES = {
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


def get_layer_structure(source_cfg: dict) -> str:
    return source_cfg.get("dataset", {}).get("layer_structure", "monthly_climatology")


def get_source_resolution(source_cfg: dict) -> str:
    return source_cfg["processing"]["source_resolution"]

def get_zip_specs(source_cfg: dict) -> list[dict]:
    """
    Return download/processing specifications for raw ZIP files.

    monthly_climatology:
      one ZIP per enabled variable.

    static_index_set:
      one ZIP for the whole index group, e.g. bio.

    static_single:
      one ZIP for one static variable, e.g. elev.

    monthly_time_series:
      one ZIP per variable and decade/period.
    """
    layer_structure = get_layer_structure(source_cfg)

    if layer_structure == "monthly_climatology":
        variables_cfg = source_cfg.get("variables", {})
        enabled = [
            variable
            for variable, cfg in variables_cfg.items()
            if cfg.get("enabled", False)
        ]

        if not enabled:
            raise ValueError("No enabled variables found in source config.")

        return [
            {
                "zip_variable_code": variable,
                "variable": variable,
                "period": None,
            }
            for variable in enabled
        ]

    if layer_structure == "static_index_set":
        zip_variable_code = source_cfg["dataset"]["zip_variable_code"]
        return [
            {
                "zip_variable_code": zip_variable_code,
                "variable": zip_variable_code,
                "period": None,
            }
        ]

    if layer_structure == "static_single":
        zip_variable_code = source_cfg["dataset"]["zip_variable_code"]
        return [
            {
                "zip_variable_code": zip_variable_code,
                "variable": zip_variable_code,
                "period": None,
            }
        ]

    if layer_structure == "monthly_time_series":
        variables_cfg = source_cfg.get("variables", {})
        periods = source_cfg.get("periods", [])

        enabled_variables = [
            variable
            for variable, cfg in variables_cfg.items()
            if cfg.get("enabled", False)
        ]

        if not enabled_variables:
            raise ValueError("No enabled variables found in source config.")

        if not periods:
            raise ValueError("No periods found in monthly_time_series source config.")

        specs = []
        for variable in enabled_variables:
            for period in periods:
                specs.append(
                    {
                        "zip_variable_code": variable,
                        "variable": variable,
                        "period": period,
                    }
                )

        return specs

    raise NotImplementedError(
        f"Unsupported layer_structure for ZIP specs: {layer_structure}"
    )


def build_worldclim_zip_name(source_cfg: dict, zip_spec: dict) -> str:
    """
    Examples:
      wc2.1_10m_tmin.zip
      wc2.1_10m_bio.zip
      wc2.1_10m_elev.zip
      wc2.1_cruts4.09_10m_tmin_1990-1999.zip
    """
    source_resolution = get_source_resolution(source_cfg)
    validate_worldclim_resolution(source_resolution)

    dataset_cfg = source_cfg.get("dataset", {})
    pattern = dataset_cfg.get(
        "zip_file_pattern",
        "wc2.1_{resolution}_{variable}.zip",
    )

    return pattern.format(
        resolution=source_resolution,
        variable=zip_spec["zip_variable_code"],
        period=zip_spec.get("period", ""),
    )


def build_worldclim_download_url(
    source_cfg: dict,
    zip_spec: dict,
) -> str:
    base_url = source_cfg["source"]["base_url"]
    zip_name = build_worldclim_zip_name(source_cfg, zip_spec)
    return f"{base_url.rstrip('/')}/{zip_name}"


def build_worldclim_zip_path(
    raw_dir: Path,
    source_cfg: dict,
    zip_spec: dict,
) -> Path:
    zip_name = build_worldclim_zip_name(source_cfg, zip_spec)
    return raw_dir / zip_name


def build_worldclim_monthly_member_basename(
    source_cfg: dict,
    variable: str,
    month: int,
) -> str:
    """
    Example:
      wc2.1_30s_tmin_01.tif
    """
    source_resolution = get_source_resolution(source_cfg)
    pattern = source_cfg["dataset"].get(
        "tif_file_pattern",
        "wc2.1_{resolution}_{variable}_{month:02d}.tif",
    )

    return pattern.format(
        resolution=source_resolution,
        variable=variable,
        month=month,
    )


def build_worldclim_static_index_member_basename(
    source_cfg: dict,
    index_number: int,
) -> str:
    """
    Example:
      wc2.1_30s_bio_1.tif
    """
    source_resolution = get_source_resolution(source_cfg)
    pattern = source_cfg["dataset"].get(
        "tif_file_pattern",
        "wc2.1_{resolution}_bio_{index}.tif",
    )

    return pattern.format(
        resolution=source_resolution,
        index=index_number,
    )


def build_worldclim_static_single_member_basename(
    source_cfg: dict,
    variable: str,
) -> str:
    """
    Example:
      wc2.1_30s_elev.tif
    """
    source_resolution = get_source_resolution(source_cfg)
    pattern = source_cfg["dataset"].get(
        "tif_file_pattern",
        "wc2.1_{resolution}_{variable}.tif",
    )

    return pattern.format(
        resolution=source_resolution,
        variable=variable,
    )

def build_worldclim_monthly_time_series_member_basename(
    source_cfg: dict,
    variable: str,
    year: int,
    month: int,
) -> str:
    """
    Example expected pattern:
      wc2.1_cruts4.09_10m_tmin_1991-01.tif
    """
    source_resolution = get_source_resolution(source_cfg)
    pattern = source_cfg["dataset"].get(
        "tif_file_pattern",
        "wc2.1_cruts4.09_{resolution}_{variable}_{year}-{month:02d}.tif",
    )

    return pattern.format(
        resolution=source_resolution,
        variable=variable,
        year=year,
        month=month,
    )


def build_worldclim_clipped_name(
    source_cfg: dict,
    layer_name: str,
    domain_name: str,
    month: int | None = None,
    year: int | None = None,
) -> str:
    """
    monthly_climatology:
      wc2.1_10m_tmin_01_pyrenees_full.tif

    monthly_time_series:
      wc2.1_cruts4.09_10m_tmin_1991_01_pyrenees_full.tif

    static index:
      wc2.1_10m_bio1_pyrenees_full.tif

    static single:
      wc2.1_10m_elev_pyrenees_full.tif
    """
    source_resolution = get_source_resolution(source_cfg)
    layer_structure = get_layer_structure(source_cfg)

    if layer_structure == "monthly_climatology":
        if month is None:
            raise ValueError("month is required for monthly_climatology clipped names")

        return (
            f"wc2.1_{source_resolution}_{layer_name}_"
            f"{month:02d}_{domain_name}.tif"
        )

    if layer_structure == "monthly_time_series":
        if year is None or month is None:
            raise ValueError(
                "year and month are required for monthly_time_series clipped names"
            )

        return (
            f"wc2.1_cruts4.09_{source_resolution}_{layer_name}_"
            f"{year}_{month:02d}_{domain_name}.tif"
        )

    return f"wc2.1_{source_resolution}_{layer_name}_{domain_name}.tif"


def build_worldclim_feature_name(
    provider: str,
    product: str,
    variable: str,
    metric: str | None,
    start_month: int | None,
    end_month: int | None,
    domain_name: str,
    target_resolution_m: int,
    start_year: int | None = None,
    end_year: int | None = None,
) -> str:
    """
    monthly climatology:
      worldclim_v2_1_climate_normals_tmin_mean_m01-m12_experimental_pallars_sobira_100m.tif

    monthly time series:
      worldclim_cruts4_09_monthly_tmin_mean_y1991-y2020_m01-m12_experimental_pallars_sobira_100m.tif

    static:
      worldclim_v2_1_bioclim_bio1_experimental_pallars_sobira_100m.tif
    """
    if (
        metric is not None
        and start_year is not None
        and end_year is not None
        and start_month is not None
        and end_month is not None
    ):
        return (
            f"{provider}_{product}_{variable}_{metric}_"
            f"y{start_year}-y{end_year}_"
            f"m{start_month:02d}-m{end_month:02d}_"
            f"{domain_name}_{int(target_resolution_m)}m.tif"
        )

    if metric is not None and start_month is not None and end_month is not None:
        return (
            f"{provider}_{product}_{variable}_{metric}_"
            f"m{start_month:02d}-m{end_month:02d}_"
            f"{domain_name}_{int(target_resolution_m)}m.tif"
        )

    return (
        f"{provider}_{product}_{variable}_"
        f"{domain_name}_{int(target_resolution_m)}m.tif"
    )