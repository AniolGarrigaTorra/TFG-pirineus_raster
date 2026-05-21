from pathlib import Path

from src.io.config import resolve_path


def build_resolution_suffix(resolution_m: int) -> str:
    return f"{int(resolution_m)}m"


def build_output_grid_suffix(project_cfg: dict, resolution_m: int) -> str:
    suffix = build_resolution_suffix(resolution_m)
    crs_suffix = project_cfg.get("_grid_crs_suffix")
    if crs_suffix:
        return f"{suffix}_{crs_suffix}"
    return suffix


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_base_dir(project_cfg: dict) -> Path | None:
    project_config_path = project_cfg.get("_config_path")
    if project_config_path is None:
        return None

    config_path = Path(project_config_path)
    if config_path.parent.name == "configs":
        return config_path.parent.parent

    return config_path.parent


def get_project_path(project_cfg: dict, key: str) -> Path:
    return resolve_path(
        project_cfg["paths"][key],
        base_path=get_project_base_dir(project_cfg),
    )


def get_grid_dir(project_cfg: dict) -> Path:
    interim_dir = get_project_path(project_cfg, "interim_dir")
    grids_subdir = project_cfg["grids"]["subdir"]
    return interim_dir / grids_subdir


def get_grid_path(project_cfg: dict, aoi_cfg: dict, resolution_m: int) -> Path:
    grid_dir = get_grid_dir(project_cfg)
    aoi_name = aoi_cfg["name"]
    resolution_suffix = build_resolution_suffix(resolution_m)
    crs_suffix = project_cfg.get("_grid_crs_suffix")
    if crs_suffix:
        return grid_dir / f"grid_{aoi_name}_{resolution_suffix}_{crs_suffix}.tif"
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
    raw_dir = get_project_path(project_cfg, "raw_dir")
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
    interim_dir = get_project_path(project_cfg, "interim_dir")
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
    processed_dir = get_project_path(project_cfg, "processed_dir")
    resolution_suffix = build_output_grid_suffix(project_cfg, target_resolution_m)

    return (
        processed_dir
        / "features"
        / provider
        / product
        / domain_name
        / resolution_suffix
    )
