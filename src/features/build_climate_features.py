from pathlib import Path
import zipfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.io.config import load_yaml


RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_grid_path(project_cfg: dict, aoi_cfg: dict) -> Path:
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    grid_subdir = project_cfg["grid"]["subdir"]
    resolution_suffix = project_cfg["naming"]["resolution_suffix"]
    aoi_name = aoi_cfg["name"]

    return interim_dir / grid_subdir / f"grid_base_{aoi_name}_{resolution_suffix}.tif"


def extract_zip(zip_path: Path, extract_root: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    target_dir = extract_root / zip_path.stem
    if not target_dir.exists():
        ensure_dir(target_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    return target_dir


def find_month_file(folder: Path, variable_name: str, month: int) -> Path:
    patterns = [
        f"*{variable_name}*_{month:02d}.tif",
        f"*{variable_name}*{month:02d}.tif",
    ]

    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Monthly file not found for variable='{variable_name}', month={month}, folder='{folder}'"
    )


def align_raster_to_grid(
    src_path: Path,
    grid_profile: dict,
    resampling_method: Resampling,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        destination = np.full(
            (grid_profile["height"], grid_profile["width"]),
            grid_profile["nodata"],
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=grid_profile["transform"],
            dst_crs=grid_profile["crs"],
            dst_nodata=grid_profile["nodata"],
            resampling=resampling_method,
        )

    return destination


def save_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        dtype="float32",
        count=1,
        compress="lzw",
    )

    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def build_valid_mask(monthly_stacks: dict[str, np.ndarray], nodata: float) -> np.ndarray:
    valid_mask = None

    for stack in monthly_stacks.values():
        current_valid = (stack != nodata).all(axis=0)
        valid_mask = current_valid if valid_mask is None else (valid_mask & current_valid)

    return valid_mask


def compute_isothermality(tmin: np.ndarray, tmax: np.ndarray, valid_mask: np.ndarray, nodata: float) -> np.ndarray:
    diurnal_range = tmax - tmin
    bio2 = diurnal_range.mean(axis=0)

    bio7 = tmax.max(axis=0) - tmin.min(axis=0)

    output = np.full_like(bio2, nodata, dtype=np.float32)
    ok = valid_mask & (bio7 != 0)
    output[ok] = (bio2[ok] / bio7[ok]) * 100.0

    return output


def compute_temp_seasonality(tavg: np.ndarray, valid_mask: np.ndarray, nodata: float) -> np.ndarray:
    output = np.full(tavg.shape[1:], nodata, dtype=np.float32)
    output[valid_mask] = tavg[:, valid_mask].std(axis=0) * 100.0
    return output


def compute_precip_sum(prec: np.ndarray, valid_mask: np.ndarray, nodata: float) -> np.ndarray:
    output = np.full(prec.shape[1:], nodata, dtype=np.float32)
    output[valid_mask] = prec[:, valid_mask].sum(axis=0)
    return output


def build_monthly_output_name(variable: str, month: int, resolution_suffix: str) -> str:
    return f"climate_{variable}_{month:02d}_{resolution_suffix}.tif"


def main():
    project_cfg = load_yaml("configs/project.yaml")
    aoi_cfg = load_yaml("configs/aoi/experimental_pallars_sobira.yaml")
    climate_cfg = load_yaml("configs/layers/climate.yaml")

    project_crs = project_cfg["crs"]
    aoi_crs = aoi_cfg["crs"]
    if project_crs != aoi_crs:
        raise ValueError(f"Project CRS ({project_crs}) does not match AOI CRS ({aoi_crs})")

    nodata = float(project_cfg["nodata"])
    resolution_suffix = project_cfg["naming"]["resolution_suffix"]

    raw_dir = Path(project_cfg["paths"]["raw_dir"])
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    processed_dir = Path(project_cfg["paths"]["processed_dir"])

    aligned_dir = interim_dir / project_cfg["alignment"]["interim_subdir"] / "climate"
    climate_out_dir = processed_dir / "climate"

    ensure_dir(aligned_dir)
    ensure_dir(climate_out_dir)

    grid_path = get_grid_path(project_cfg, aoi_cfg)
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid file not found: {grid_path}\n"
            f"Run src/make_grid.py first."
        )

    with rasterio.open(grid_path) as grid:
        grid_profile = grid.profile.copy()
        grid_profile["nodata"] = nodata

    climate_source_raw_dir = Path(climate_cfg["source"]["raw_dir"])
    if not climate_source_raw_dir.is_absolute():
        climate_source_raw_dir = climate_source_raw_dir

    months = climate_cfg["aggregation"]["months"]
    variables_cfg = climate_cfg["variables"]

    resampling_name = climate_cfg["resampling"]["continuous"]
    resampling_method = RESAMPLING_MAP[resampling_name]

    extracted_dirs = {}
    monthly_arrays = {var_name: [] for var_name in variables_cfg.keys()}

    for var_name, var_cfg in variables_cfg.items():
        zip_path = Path(var_cfg["zip_file"])
        if not zip_path.is_absolute():
            zip_path = climate_source_raw_dir / zip_path

        extracted_dirs[var_name] = extract_zip(zip_path, climate_source_raw_dir)

    for var_name, var_cfg in variables_cfg.items():
        scale_factor = float(var_cfg.get("scale_factor", 1.0))
        source_folder = extracted_dirs[var_name]

        for month in months:
            month_file = find_month_file(source_folder, var_name, month)
            aligned_array = align_raster_to_grid(
                src_path=month_file,
                grid_profile=grid_profile,
                resampling_method=resampling_method,
            )

            valid = aligned_array != nodata
            aligned_array[valid] = aligned_array[valid] * scale_factor

            monthly_arrays[var_name].append(aligned_array)

            monthly_output_path = aligned_dir / build_monthly_output_name(
                variable=var_name,
                month=month,
                resolution_suffix=resolution_suffix,
            )
            save_raster(monthly_output_path, aligned_array, grid_profile)

    monthly_stacks = {
        var_name: np.stack(arrays, axis=0)
        for var_name, arrays in monthly_arrays.items()
    }

    valid_mask = build_valid_mask(monthly_stacks, nodata)

    isothermality = compute_isothermality(
        tmin=monthly_stacks["tmin"],
        tmax=monthly_stacks["tmax"],
        valid_mask=valid_mask,
        nodata=nodata,
    )

    temp_seasonality = compute_temp_seasonality(
        tavg=monthly_stacks["tavg"],
        valid_mask=valid_mask,
        nodata=nodata,
    )

    precip_sum = compute_precip_sum(
        prec=monthly_stacks["prec"],
        valid_mask=valid_mask,
        nodata=nodata,
    )

    derived_cfg = climate_cfg["derived_layers"]

    save_raster(
        climate_out_dir / derived_cfg["isothermality"]["output_name"],
        isothermality,
        grid_profile,
    )

    save_raster(
        climate_out_dir / derived_cfg["temp_seasonality"]["output_name"],
        temp_seasonality,
        grid_profile,
    )

    save_raster(
        climate_out_dir / derived_cfg["precip_sum"]["output_name"],
        precip_sum,
        grid_profile,
    )

    print("Climate features created successfully")
    print(f"  Grid: {grid_path}")
    print(f"  Monthly aligned rasters: {aligned_dir}")
    print(f"  Final climate rasters: {climate_out_dir}")


if __name__ == "__main__":
    main()