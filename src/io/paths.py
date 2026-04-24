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


def get_aligned_dir(project_cfg: dict, category: str, resolution_m: int) -> Path:
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    aligned_subdir = project_cfg["alignment"]["interim_subdir"]
    resolution_suffix = build_resolution_suffix(resolution_m)
    return interim_dir / aligned_subdir / category / resolution_suffix


def get_processed_dir(project_cfg: dict, category: str, resolution_m: int) -> Path:
    processed_dir = Path(project_cfg["paths"]["processed_dir"])
    resolution_suffix = build_resolution_suffix(resolution_m)
    return processed_dir / category / resolution_suffix