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


def get_zip_variable_codes(source_cfg: dict) -> list[str]:
    """
    Return the raw ZIP variable codes that must be downloaded.

    monthly_climatology:
      one ZIP per enabled variable:
      wc2.1_30s_tmin.zip
      wc2.1_30s_prec.zip

    static_index_set:
      one ZIP for the index group:
      wc2.1_30s_bio.zip

    static_single:
      one ZIP for the variable:
      wc2.1_30s_elev.zip
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

        return enabled

    if layer_structure in {"static_index_set", "static_single"}:
        zip_variable_code = source_cfg["dataset"]["zip_variable_code"]
        return [zip_variable_code]

    raise NotImplementedError(
        f"Unsupported layer_structure for ZIP naming: {layer_structure}"
    )


def build_worldclim_zip_name(source_cfg: dict, zip_variable_code: str) -> str:
    """
    Examples:
      wc2.1_30s_tmin.zip
      wc2.1_30s_bio.zip
      wc2.1_30s_elev.zip
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
        variable=zip_variable_code,
    )


def build_worldclim_download_url(
    source_cfg: dict,
    zip_variable_code: str,
) -> str:
    base_url = source_cfg["source"]["base_url"]
    zip_name = build_worldclim_zip_name(source_cfg, zip_variable_code)
    return f"{base_url.rstrip('/')}/{zip_name}"


def build_worldclim_zip_path(
    raw_dir: Path,
    source_cfg: dict,
    zip_variable_code: str,
) -> Path:
    zip_name = build_worldclim_zip_name(source_cfg, zip_variable_code)
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


def build_worldclim_clipped_name(
    source_cfg: dict,
    layer_name: str,
    domain_name: str,
    month: int | None = None,
) -> str:
    """
    Output clipped intermediate name.

    monthly:
      wc2.1_30s_tmin_01_pyrenees_full.tif

    static index:
      wc2.1_30s_bio1_pyrenees_full.tif

    static single:
      wc2.1_30s_elev_pyrenees_full.tif
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
) -> str:
    """
    monthly example:
      worldclim_v2_1_climate_normals_tmin_mean_m01-m12_experimental_pallars_sobira_100m.tif

    static example:
      worldclim_v2_1_bioclim_bio1_experimental_pallars_sobira_100m.tif
    """
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