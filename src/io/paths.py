from pathlib import Path


def build_resolution_suffix(resolution_m: int) -> str:
    return f"{int(resolution_m)}m"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_grid_dir(project_cfg: dict) -> Path:
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    grids_subdir = project_cfg["grids"]["subdir"]
    return interim_dir / grids_subdir


def get_grid_path(project_cfg: dict, aoi_cfg: dict, resolution_m: int) -> Path:
    grid_dir = get_grid_dir(project_cfg)
    aoi_name = aoi_cfg["name"]
    resolution_suffix = build_resolution_suffix(resolution_m)
    return grid_dir / f"grid_{aoi_name}_{resolution_suffix}.tif"

def get_source_raw_dir(
    project_cfg: dict,
    provider: str,
    product: str,
    source_resolution: str,
) -> Path:
    """
    Raw global source files.

    Example:
    data_raw/worldclim/v2_1_base/10m/
    """
    raw_dir = Path(project_cfg["paths"]["raw_dir"])
    return raw_dir / provider / product / source_resolution


def get_source_interim_dir(
    project_cfg: dict,
    provider: str,
    product: str,
) -> Path:
    """
    Intermediate source data.

    Example:
    data_interim/sources/worldclim/v2_1_base/
    """
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    return interim_dir / "sources" / provider / product


def get_source_clipped_dir(
    project_cfg: dict,
    provider: str,
    product: str,
    domain_name: str,
    source_resolution: str,
    variable: str,
) -> Path:
    """
    Clipped monthly rasters.

    Example:
    data_interim/sources/worldclim/v2_1_base/clipped/experimental_pallars_sobira/10m/tmin/
    """
    return (
        get_source_interim_dir(project_cfg, provider, product)
        / "clipped"
        / domain_name
        / source_resolution
        / variable
    )


def get_feature_output_dir(
    project_cfg: dict,
    provider: str,
    product: str,
    domain_name: str,
    target_resolution_m: int,
) -> Path:
    """
    Final processed feature rasters.

    Example:
    data_processed/features/worldclim/v2_1_base/experimental_pallars_sobira/100m/
    """
    processed_dir = Path(project_cfg["paths"]["processed_dir"])
    resolution_suffix = build_resolution_suffix(target_resolution_m)

    return (
        processed_dir
        / "features"
        / provider
        / product
        / domain_name
        / resolution_suffix
    )