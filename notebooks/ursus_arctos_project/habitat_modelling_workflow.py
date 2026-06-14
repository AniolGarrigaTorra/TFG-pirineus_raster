"""Reusable workflow helpers for the brown-bear habitat modelling notebook.

The notebook remains the user-facing analysis. This module contains the mechanics that
benefit from testing and reuse: run provenance, biological-data treatment, background
pool isolation, model evaluation, calibration, and spatial products.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import rowcol, xy
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr
from shapely.geometry import MultiPoint, Point, box, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold

try:
    import joblib
except ImportError:  # pragma: no cover - exercised only in minimal environments
    joblib = None

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    import xgboost
except ImportError:  # pragma: no cover
    xgboost = None


CUB_CLASSES = ("lt6", "6to12")
SEASON_ORDER = ("winter", "spring", "summer", "autumn")
SEASON_MONTHS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "autumn": {9, 10, 11},
}
CUB_CLASS_MAP = {"<6month": "lt6", "1": "6to12", "1.0": "6to12"}
REDUCED_FEATURE_EXCLUSIONS = {
    "topography_roughness",
    "topography_slope",
    "climate_max_temperature",
    "climate_min_temperature",
    "habitat_tree_cover_density",
}
SHORT_FEATURE_LABELS = {
    "climate_precipitation_annual": "Annual precipitation",
    "climate_solar_radiation_annual": "Annual solar radiation",
    "climate_mean_temperature": "Mean temperature",
    "climate_snow_cover": "Snow cover",
    "climate_water_availability": "Water availability",
    "topography_dem_elevation": "Elevation",
    "topography_ruggedness": "Ruggedness",
    "topography_tpi_macro": "Broad-scale TPI",
    "topography_tpi_med": "Medium-scale TPI",
    "topography_aspect_sin": "Aspect (north-south)",
    "topography_aspect_cos": "Aspect (east-west)",
    "human_logdistance_tracks": "Distance to tracks",
    "human_logdistance_towns": "Distance to towns",
    "human_logdistance_secondary_roads_transport_secondary_roads_distance": "Distance to secondary roads",
    "human_population_density_mean": "Population density",
    "human_landuse_presence": "Human land use",
    "habitat_biomass_mean": "Biomass",
    "habitat_productivity_mean": "Productivity",
    "habitat_broadleaf": "Broadleaf forest",
    "habitat_coniferous_dominant_leaf_type": "Coniferous forest",
    "habitat_mix_forest_habitat_broadleaf_habitat_coniferous_dominant_leaf_type": "Mixed forest",
    "habitat_grasslands": "Grassland",
    "habitat_shrubland_2018": "Shrubland",
    "habitat_screes_and_bare_rocks_2018": "Scree and bare rock",
}


def coerce_bool(value: Any) -> bool:
    """Normalize booleans passed through Papermill's string-valued CLI parameters."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


@dataclass
class WorkflowConfig:
    output_dir: str = "outputs/ursus_arctos_habitat_modelling/run_2"
    random_seed: int = 42
    n_background_train: int = 10_000
    n_background_test: int = 2_000
    min_thin_distance_m: int = 100
    n_thin_iterations: int = 20
    local_buffer_km: float = 25.0
    local_buffer_sensitivity_km: tuple[float, ...] = (20.0, 25.0, 30.0)
    custom_local_domain_path: str | None = None
    n_final_bootstrap_replicates: int = 10
    n_evaluation_replicates: int = 5
    n_paper_validation_repeats: int = 10
    n_rf_tuning_iter: int = 12
    n_optuna_trials: int = 20
    inner_blocks_axis: int = 2
    run_tuning: bool = True
    strict_nested_tuning: bool = True
    run_xgboost: bool = True
    run_shap: bool = True
    run_map_prediction: bool = True
    smoke_mode: bool = False
    batch_size: int = 100_000
    prediction_season_by_class: dict[str, str] = field(
        default_factory=lambda: {"lt6": "spring", "6to12": "summer"}
    )
    raster_dir: str = "data_processed/datasets/ursus_arctos_pyrenees_100m/rasters"
    gps_path: str = "data_processed/notebooks/ursus_arctos_project/GPS_female_bears_with_cubs.csv"
    obs_path: str = "data_processed/notebooks/ursus_arctos_project/observations_female_bears_with_cubs.csv"
    individuals_path: str = "data_processed/notebooks/ursus_arctos_project/bear_individuals.csv"
    grid_path: str = "data_interim/grids/grid_ursus_arctos_pyrenees_100m.tif"
    aoi_config_path: str = "configs/aoi/ursus_arctos_pyrenees.yaml"
    run_config_path: str = "configs/runs/ursus_arctos_pyrenees_100m.yaml"
    maxent_reference_dir: str = "use_cases_info"

    def __post_init__(self):
        for name in (
            "run_tuning",
            "strict_nested_tuning",
            "run_xgboost",
            "run_shap",
            "run_map_prediction",
            "smoke_mode",
        ):
            setattr(self, name, coerce_bool(getattr(self, name)))

    def effective(self) -> "WorkflowConfig":
        if not self.smoke_mode:
            return self
        values = asdict(self)
        values.update(
            n_background_train=min(self.n_background_train, 1_000),
            n_background_test=min(self.n_background_test, 300),
            n_final_bootstrap_replicates=min(self.n_final_bootstrap_replicates, 2),
            n_evaluation_replicates=1,
            n_paper_validation_repeats=2,
            n_rf_tuning_iter=min(self.n_rf_tuning_iter, 2),
            n_optuna_trials=min(self.n_optuna_trials, 2),
            run_shap=False,
            run_map_prediction=False,
        )
        values["local_buffer_sensitivity_km"] = tuple(values["local_buffer_sensitivity_km"])
        return WorkflowConfig(**values)


@dataclass
class RunPaths:
    root: Path
    tables: Path
    plots: Path
    maps: Path
    models: Path
    logs: Path
    intermediate: Path

    @classmethod
    def create(cls, root: Path) -> "RunPaths":
        result = cls(
            root=root,
            tables=root / "tables",
            plots=root / "plots",
            maps=root / "maps",
            models=root / "models",
            logs=root / "logs",
            intermediate=root / "intermediate",
        )
        for path in asdict(result).values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return result


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    raise FileNotFoundError("Could not find the pirineus-raster project root.")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ursus_arctos_habitat")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def safe_variable_name(name: Any) -> str:
    cleaned = str(name).strip().replace(" ", "_").replace("-", "_")
    return "_".join(part for part in cleaned.split("_") if part).lower()


def feature_name_from_path(path: Path) -> str:
    name = path.stem
    for suffix in ("_ursus_arctos_pyrenees_100m", "_pyrenees_100m", "_100m"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def infer_temporal_metadata(variable: str) -> dict[str, Any]:
    variable = safe_variable_name(variable)
    tokens = variable.split("_")
    season = next((item for item in SEASON_ORDER if item in tokens), None)
    if variable.startswith("climate_snow_cover_") and season:
        return {"model_variable": "climate_snow_cover", "temporal_dimension": season, "temporal_role": "seasonal"}
    if season and tokens[-1] == season:
        return {"model_variable": "_".join(tokens[:-1]), "temporal_dimension": season, "temporal_role": "seasonal"}
    if tokens[-1] == "annual":
        return {"model_variable": variable, "temporal_dimension": "annual", "temporal_role": "annual"}
    return {"model_variable": variable, "temporal_dimension": None, "temporal_role": "static"}


def load_raster_inventory(raster_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted([*raster_dir.glob("*.tif"), *raster_dir.glob("*.tiff")]):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        variable = safe_variable_name(metadata.get("variable") or feature_name_from_path(path))
        temporal = infer_temporal_metadata(variable)
        with rasterio.open(path) as src:
            records.append(
                {
                    "path": str(path),
                    "variable": variable,
                    **temporal,
                    "unit": metadata.get("unit"),
                    "description": metadata.get("variable_description") or metadata.get("description"),
                    "crs": str(src.crs),
                    "width": src.width,
                    "height": src.height,
                    "res_x": src.res[0],
                    "res_y": src.res[1],
                    "nodata": src.nodata,
                    "dtype": src.dtypes[0],
                    "modified_utc": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True),
                }
            )
    if not records:
        raise FileNotFoundError(f"No environmental rasters found in {raster_dir}")
    return pd.DataFrame(records)


def check_alignment(inventory: pd.DataFrame, grid_path: Path) -> pd.DataFrame:
    records = []
    with rasterio.open(grid_path) as grid:
        for item in inventory.itertuples(index=False):
            with rasterio.open(item.path) as src:
                records.append(
                    {
                        "variable": item.variable,
                        "same_shape": src.shape == grid.shape,
                        "same_transform": src.transform.almost_equals(grid.transform),
                        "same_bounds": np.allclose(tuple(src.bounds), tuple(grid.bounds), atol=1e-6),
                        "same_crs": src.crs == grid.crs,
                    }
                )
    return pd.DataFrame(records)


def profile_environment(inventory: pd.DataFrame) -> pd.DataFrame:
    records = []
    for item in inventory.itertuples(index=False):
        with rasterio.open(item.path) as src:
            values = src.read(1, masked=True).compressed().astype(float)
            total = src.width * src.height
        valid = values[np.isfinite(values)]
        valid_pct = 100 * valid.size / total
        records.append(
            {
                "variable": item.variable,
                "model_variable": item.model_variable,
                "temporal_role": item.temporal_role,
                "temporal_dimension": item.temporal_dimension,
                "valid_count": valid.size,
                "total_pixels": total,
                "valid_pct": valid_pct,
                "nodata_pct": 100 - valid_pct,
                "min": float(np.min(valid)) if valid.size else np.nan,
                "max": float(np.max(valid)) if valid.size else np.nan,
                "mean": float(np.mean(valid)) if valid.size else np.nan,
                "std": float(np.std(valid)) if valid.size else np.nan,
                "zero_pct": float(100 * np.mean(valid == 0)) if valid.size else np.nan,
                "issue": "ok" if valid_pct >= 90 and valid.size and np.std(valid) > 0 else "valid_coverage_below_90pct",
                "path": item.path,
            }
        )
    return pd.DataFrame(records)


def build_common_mask(inventory: pd.DataFrame, grid_path: Path, output_path: Path) -> np.ndarray:
    with rasterio.open(grid_path) as grid:
        common = np.ones(grid.shape, dtype=bool)
        profile = grid.profile.copy()
    for item in inventory.itertuples(index=False):
        with rasterio.open(item.path) as src:
            data = src.read(1, masked=True)
            common &= ~np.ma.getmaskarray(data) & np.isfinite(data.filled(np.nan))
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(common.astype("uint8"), 1)
    return common


def parse_mixed_datetime(values: pd.Series) -> pd.Series:
    """Parse mixed European and ISO dates without discarding valid records."""
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce", format="mixed", yearfirst=True, utc=True)
    european = ~iso & text.notna()
    if european.any():
        parsed.loc[european] = pd.to_datetime(text.loc[european], errors="coerce", format="mixed", dayfirst=True, utc=True)
    return parsed


def recode_cub_class(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().map(CUB_CLASS_MAP).astype("string")


def season_from_month(month: Any) -> str | None:
    if pd.isna(month):
        return None
    return next((season for season, months in SEASON_MONTHS.items() if int(month) in months), None)


def add_event_fields(frame: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["event_month"] = result[datetime_col].dt.month
    result["event_season"] = result["event_month"].map(season_from_month).astype("string")
    result["event_day"] = result[datetime_col].dt.floor("D")
    bear = result.get("bear_name", pd.Series("unknown", index=result.index)).astype("string")
    result["daily_group"] = bear.fillna("unknown") + "_" + result["event_day"].astype("string")
    return result


def reproject_points(frame: pd.DataFrame, target_crs: str) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x, y = transformer.transform(
        pd.to_numeric(frame["x_long"], errors="coerce").to_numpy(),
        pd.to_numeric(frame["y_lat"], errors="coerce").to_numpy(),
    )
    result = frame.copy()
    result["x_3035"] = x
    result["y_3035"] = y
    return result


def bounds_filter(frame: pd.DataFrame, bounds: dict[str, float]) -> pd.Series:
    lon = pd.to_numeric(frame["x_long"], errors="coerce")
    lat = pd.to_numeric(frame["y_lat"], errors="coerce")
    return lon.between(bounds["xmin"], bounds["xmax"]) & lat.between(bounds["ymin"], bounds["ymax"])


def clean_biological_data(
    gps_path: Path,
    obs_path: Path,
    individuals_path: Path,
    target_crs: str,
    bounds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gps = pd.read_csv(gps_path)
    obs = pd.read_csv(obs_path)
    individuals = pd.read_csv(individuals_path)
    gps_report: list[dict[str, Any]] = []
    obs_report: list[dict[str, Any]] = []

    def apply(frame: pd.DataFrame, mask: pd.Series, step: str, report: list[dict[str, Any]]) -> pd.DataFrame:
        before = len(frame)
        result = frame.loc[mask].copy()
        report.append({"step": step, "before": before, "after": len(result), "removed": before - len(result)})
        return result

    before = len(gps)
    gps = gps.drop_duplicates("id_obs").copy()
    gps_report.append({"step": "drop duplicated id_obs", "before": before, "after": len(gps), "removed": before - len(gps)})
    gps["datetime_utc"] = parse_mixed_datetime(
        gps["date_gmt"].astype("string").fillna("") + " " + gps["time_gmt"].astype("string").fillna("00:00:00")
    )
    gps["cub_class"] = recode_cub_class(gps["with_cubs_estimated"])
    gps = add_event_fields(gps, "datetime_utc")
    gps["event_year"] = gps["datetime_utc"].dt.year
    gps = apply(gps, gps["cub_class"].notna(), "keep target cub classes", gps_report)
    gps = apply(gps, bounds_filter(gps, bounds), "keep points inside AOI lon/lat bounds", gps_report)
    gps = apply(gps, gps["event_season"].notna(), "keep points with parseable event season", gps_report)
    registry = individuals[["name", "mortality_year", "suposed_desaparition_year"]].copy()
    registry["name_key"] = registry["name"].astype("string").str.lower().str.strip()
    gps["name_key"] = gps["bear_name"].astype("string").str.lower().str.strip()
    gps = gps.merge(registry, on="name_key", how="left", suffixes=("", "_registry"))
    gps["end_year"] = gps[["mortality_year", "suposed_desaparition_year"]].min(axis=1, skipna=True)
    valid_year = ~(gps["end_year"].notna() & gps["event_year"].notna() & (gps["event_year"] > gps["end_year"]))
    gps = apply(gps, valid_year, "remove records after mortality/disappearance year", gps_report)

    before = len(obs)
    obs = obs.drop_duplicates("id_obs").copy()
    obs_report.append({"step": "drop duplicated id_obs", "before": before, "after": len(obs), "removed": before - len(obs)})
    obs["date_parsed"] = parse_mixed_datetime(obs["date"])
    obs["cub_class"] = recode_cub_class(obs["with_cubs_estimated"])
    obs = add_event_fields(obs, "date_parsed")
    obs = apply(obs, obs["cub_class"].notna(), "keep target cub classes", obs_report)
    obs = apply(obs, bounds_filter(obs, bounds), "keep points inside AOI lon/lat bounds", obs_report)
    failed_dates = obs.loc[obs["event_season"].isna(), ["id_obs", "date", "cub_class", "x_long", "y_lat"]].copy()
    obs = apply(obs, obs["event_season"].notna(), "keep points with parseable event season", obs_report)
    return (
        reproject_points(gps, target_crs).reset_index(drop=True),
        reproject_points(obs, target_crs).reset_index(drop=True),
        pd.DataFrame(gps_report),
        pd.DataFrame(obs_report),
        failed_dates,
    )


def assign_grid_cells(points: pd.DataFrame, grid_path: Path) -> pd.DataFrame:
    result = points.copy()
    with rasterio.open(grid_path) as grid:
        rows, cols = rowcol(grid.transform, result["x_3035"].to_numpy(), result["y_3035"].to_numpy())
        result["grid_row"] = np.asarray(rows, dtype=int)
        result["grid_col"] = np.asarray(cols, dtype=int)
        inside = result["grid_row"].between(0, grid.height - 1) & result["grid_col"].between(0, grid.width - 1)
    return result.loc[inside].copy()


def spatial_thin_one(frame: pd.DataFrame, min_dist_m: float, iterations: int, seed: int) -> pd.DataFrame:
    if len(frame) <= 1 or min_dist_m <= 0:
        return frame.copy()
    coords = frame[["x_3035", "y_3035"]].to_numpy(dtype=float)
    pairs = cKDTree(coords).query_pairs(r=max(min_dist_m - 1e-9, 0))
    conflicts = [set() for _ in range(len(frame))]
    for left, right in pairs:
        conflicts[left].add(right)
        conflicts[right].add(left)
    rng = np.random.default_rng(seed)
    best = np.ones(len(frame), dtype=bool)
    best_count = -1
    for _ in range(iterations):
        blocked = np.zeros(len(frame), dtype=bool)
        keep = np.zeros(len(frame), dtype=bool)
        for index in rng.permutation(len(frame)):
            if blocked[index]:
                continue
            keep[index] = True
            if conflicts[index]:
                blocked[list(conflicts[index])] = True
        if keep.sum() > best_count:
            best, best_count = keep, int(keep.sum())
    return frame.iloc[np.flatnonzero(best)].copy()


def treat_presences(gps: pd.DataFrame, config: WorkflowConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dedup_records = []
    thin_records = []
    parts = []
    for cub_class in CUB_CLASSES:
        class_points = gps[gps["cub_class"] == cub_class].copy()
        before = len(class_points)
        class_points = class_points.drop_duplicates(["bear_name", "event_season", "grid_row", "grid_col"])
        dedup_records.append({"cub_class": cub_class, "before": before, "after": len(class_points), "removed": before - len(class_points)})
        for offset, ((bear, season), group) in enumerate(class_points.groupby(["bear_name", "event_season"], dropna=False)):
            treated = spatial_thin_one(group, config.min_thin_distance_m, config.n_thin_iterations, config.random_seed + offset)
            thin_records.append(
                {
                    "cub_class": cub_class,
                    "bear_name": bear,
                    "event_season": season,
                    "before": len(group),
                    "after": len(treated),
                    "removed": len(group) - len(treated),
                }
            )
            parts.append(treated)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(dedup_records), pd.DataFrame(thin_records)


def load_custom_geometry(path: Path, target_crs: str):
    import geopandas as gpd

    frame = gpd.read_file(path)
    if frame.empty:
        raise ValueError(f"Custom local domain is empty: {path}")
    if frame.crs is None:
        raise ValueError("Custom local domain must declare its CRS.")
    return frame.to_crs(target_crs).geometry.union_all()


def build_domain_geometries(
    gps: pd.DataFrame,
    grid_path: Path,
    target_crs: str,
    config: WorkflowConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    with rasterio.open(grid_path) as grid:
        full = box(*grid.bounds)
    records = []
    sensitivity: dict[float, Any] = {}
    for buffer_km in config.local_buffer_sensitivity_km:
        geometry = unary_union(
            [
                MultiPoint(group[["x_3035", "y_3035"]].to_numpy()).convex_hull.buffer(buffer_km * 1_000)
                for _, group in gps.groupby("bear_name")
            ]
        ).intersection(full)
        sensitivity[float(buffer_km)] = geometry
        records.append(
            {
                "buffer_km": float(buffer_km),
                "area_km2": geometry.area / 1_000_000,
                "n_components": len(geometry.geoms) if hasattr(geometry, "geoms") else 1,
                "contains_all_gps": bool(all(geometry.covers(Point(x, y)) for x, y in gps[["x_3035", "y_3035"]].to_numpy())),
            }
        )
    if config.custom_local_domain_path:
        local = load_custom_geometry(Path(config.custom_local_domain_path), target_crs).intersection(full)
        local_source = "custom"
    else:
        local = sensitivity[float(config.local_buffer_km)]
        local_source = "separate_bear_hulls"
    if not all(local.covers(Point(x, y)) for x, y in gps[["x_3035", "y_3035"]].to_numpy()):
        raise ValueError("The selected local domain does not contain every treated GPS point.")
    table = pd.DataFrame(records)
    table["selected"] = np.isclose(table["buffer_km"], config.local_buffer_km) & (local_source != "custom")
    table["source"] = local_source
    return {"full_aoi": full, "local_domain": local}, table


def domain_mask(geometry: Any, grid_path: Path) -> np.ndarray:
    with rasterio.open(grid_path) as grid:
        return geometry_mask([mapping(geometry)], out_shape=grid.shape, transform=grid.transform, invert=True)


def sample_background_cells(
    valid_mask: np.ndarray,
    n: int,
    seed: int,
    excluded_ids: set[int] | None = None,
) -> pd.DataFrame:
    rows, cols = np.where(valid_mask)
    with rasterio.open(sample_background_cells.grid_path) as grid:  # type: ignore[attr-defined]
        width = grid.width
        ids = rows.astype(np.int64) * width + cols.astype(np.int64)
        if excluded_ids:
            keep = ~np.isin(ids, np.fromiter(excluded_ids, dtype=np.int64))
            rows, cols, ids = rows[keep], cols[keep], ids[keep]
        if n > len(rows):
            raise ValueError(f"Requested {n} unique background cells from only {len(rows)} available cells.")
        chosen = np.random.default_rng(seed).choice(len(rows), n, replace=False)
        selected_rows, selected_cols, selected_ids = rows[chosen], cols[chosen], ids[chosen]
        xs, ys = xy(grid.transform, selected_rows, selected_cols, offset="center")
    return pd.DataFrame(
        {
            "cell_id": selected_ids,
            "x_3035": np.asarray(xs, dtype=float),
            "y_3035": np.asarray(ys, dtype=float),
            "grid_row": selected_rows,
            "grid_col": selected_cols,
            "label": 0,
            "source": "background",
            "bear_name": "background",
        }
    )


def create_background_pools(
    masks: dict[str, np.ndarray],
    grid_path: Path,
    config: WorkflowConfig,
) -> dict[str, pd.DataFrame]:
    sample_background_cells.grid_path = grid_path  # type: ignore[attr-defined]
    local_test = sample_background_cells(masks["local_domain"], config.n_background_test, config.random_seed + 100)
    excluded = set(local_test["cell_id"].astype(int))
    full_test = sample_background_cells(masks["full_aoi"], config.n_background_test, config.random_seed + 200, excluded)
    excluded.update(full_test["cell_id"].astype(int))
    local_train = sample_background_cells(masks["local_domain"], config.n_background_train, config.random_seed + 300, excluded)
    excluded.update(local_train["cell_id"].astype(int))
    full_train = sample_background_cells(masks["full_aoi"], config.n_background_train, config.random_seed + 400, excluded)
    pools = {
        "local_test": local_test,
        "full_test": full_test,
        "local_train": local_train,
        "full_train": full_train,
    }
    test_ids = set(local_test.cell_id) | set(full_test.cell_id)
    train_ids = set(local_train.cell_id) | set(full_train.cell_id)
    if test_ids & train_ids:
        raise AssertionError("Training and evaluation background pools overlap.")
    return pools


def sample_raster(path: Path, coords: list[tuple[float, float]]) -> np.ndarray:
    with rasterio.open(path) as src:
        values = np.array([item[0] for item in src.sample(coords)], dtype=float)
        if src.nodata is not None:
            values[np.isclose(values, src.nodata)] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def extract_features(points: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    coords = list(zip(points["x_3035"].astype(float), points["y_3035"].astype(float)))
    values: dict[str, np.ndarray] = {}
    for item in inventory[inventory["temporal_role"].isin(["static", "annual"])].itertuples(index=False):
        values.setdefault(item.model_variable, sample_raster(Path(item.path), coords))
    for variable, group in inventory[inventory["temporal_role"] == "seasonal"].groupby("model_variable"):
        output = np.full(len(points), np.nan)
        for item in group.itertuples(index=False):
            mask = points["event_season"].astype("string").eq(str(item.temporal_dimension)).to_numpy()
            if mask.any():
                indexes = np.flatnonzero(mask)
                output[indexes] = sample_raster(Path(item.path), [coords[index] for index in indexes])
        if "snow_cover" in variable:
            no_snow = points["event_season"].isin(["summer", "autumn"]).to_numpy()
            output[np.isnan(output) & no_snow] = 0.0
        values[variable] = output
    result = pd.DataFrame(values, index=points.index)
    if "topography_aspect" in result:
        radians = np.deg2rad(pd.to_numeric(result.pop("topography_aspect"), errors="coerce"))
        result["topography_aspect_sin"] = np.sin(radians)
        result["topography_aspect_cos"] = np.cos(radians)
    return result


def assign_background_seasons(background: pd.DataFrame, presences: pd.DataFrame, seed: int) -> pd.DataFrame:
    seasons = presences["event_season"].dropna().astype(str).to_numpy()
    if not len(seasons):
        raise ValueError("Cannot assign background seasons without dated presences.")
    result = background.copy()
    result["event_season"] = np.random.default_rng(seed).choice(seasons, len(result), replace=True)
    result["daily_group"] = "background"
    return result


def presence_weights(presences: pd.DataFrame) -> np.ndarray:
    """Give equal total influence to bears and to observed days within each bear."""
    weights = pd.Series(0.0, index=presences.index)
    bears = list(presences["bear_name"].dropna().unique())
    for bear in bears:
        bear_rows = presences[presences["bear_name"] == bear]
        days = list(bear_rows["daily_group"].dropna().unique())
        for day in days:
            indexes = bear_rows.index[bear_rows["daily_group"] == day]
            weights.loc[indexes] = 1.0 / (len(bears) * len(days) * len(indexes))
    return weights.to_numpy()


def model_sample_weights(table: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(table), dtype=float)
    positive = table["label"].to_numpy() == 1
    if positive.any():
        weights[positive] = presence_weights(table.loc[positive])
    if (~positive).any():
        weights[~positive] = 1.0 / (~positive).sum()
    weights[positive] *= 0.5 / max(weights[positive].sum(), 1e-12)
    weights[~positive] *= 0.5 / max(weights[~positive].sum(), 1e-12)
    # Preserve relative ecological weights while keeping XGBoost Hessian scales
    # comparable to ordinary per-row weights.
    return weights * len(table)


def build_model_data(
    presences: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    inventory: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    all_variables = set(inventory["model_variable"])
    full_features = sorted((all_variables - {"topography_aspect"}) | {"topography_aspect_sin", "topography_aspect_cos"})
    reduced_features = [name for name in full_features if name not in REDUCED_FEATURE_EXCLUSIONS]
    result: dict[str, dict[str, Any]] = {}
    for cub_class in CUB_CLASSES:
        class_pres = presences[presences["cub_class"] == cub_class].copy()
        class_pres["point_id"] = class_pres["id_obs"].astype("string")
        class_pres["label"] = 1
        class_pres["source"] = "gps"
        presence_columns = [
            "point_id", "x_3035", "y_3035", "grid_row", "grid_col", "source", "label",
            "bear_name", "event_season", "daily_group",
        ]
        presence_table = class_pres[presence_columns].reset_index(drop=True)
        presence_features = extract_features(presence_table, inventory)
        presence_table = pd.concat([presence_table, presence_features], axis=1)
        class_data: dict[str, Any] = {"presences": presence_table}
        pool_offsets = {"local_test": 100, "full_test": 200, "local_train": 300, "full_train": 400}
        class_offset = 0 if cub_class == "lt6" else 1_000
        for pool_name, pool in pools.items():
            seasonal = assign_background_seasons(pool, class_pres, 10_000 + class_offset + pool_offsets[pool_name])
            seasonal["point_id"] = pool_name + "_" + seasonal["cell_id"].astype(str)
            columns = [
                "point_id", "cell_id", "x_3035", "y_3035", "grid_row", "grid_col", "source",
                "label", "bear_name", "event_season", "daily_group",
            ]
            table = seasonal[columns].reset_index(drop=True)
            table = pd.concat([table, extract_features(table, inventory)], axis=1)
            class_data[pool_name] = table
        for key, table in class_data.items():
            missing = table[reduced_features].isna().any(axis=1)
            if missing.any():
                class_data[key] = table.loc[~missing].reset_index(drop=True)
        result[cub_class] = class_data
    return result, reduced_features


DEFAULT_RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": 42,
}
DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}
RF_PARAM_DIST = {
    "n_estimators": [300, 500, 800, 1_000],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "min_samples_leaf": [2, 5, 10, 20],
}


def boyce_index(scores: Iterable[float], presence_mask: Iterable[bool], n_bins: int = 10) -> float:
    scores_array = np.asarray(scores, dtype=float)
    presence = np.asarray(presence_mask, dtype=bool)
    valid = np.isfinite(scores_array)
    scores_array, presence = scores_array[valid], presence[valid]
    if presence.sum() < 2 or (~presence).sum() < 2 or np.unique(scores_array).size < 3:
        return np.nan
    edges = np.linspace(scores_array.min(), scores_array.max(), n_bins + 1)
    midpoints, ratios = [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        inside = (scores_array >= lower) & (scores_array <= upper if upper == edges[-1] else scores_array < upper)
        expected = inside.mean()
        observed = inside[presence].mean()
        if expected > 0:
            midpoints.append((lower + upper) / 2)
            ratios.append(observed / expected)
    return float(spearmanr(midpoints, ratios).correlation) if len(ratios) >= 3 else np.nan


def evaluate_predictions(y_true: Iterable[int], scores: Iterable[float], sample_weight: Iterable[float] | None = None) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(scores, dtype=float)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, probability, sample_weight=weight)
    best = int(np.nanargmax(tpr - fpr))
    return {
        "auc_roc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "auc_pr": float(average_precision_score(y, probability, sample_weight=weight)),
        "boyce": boyce_index(probability, y == 1),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1], sample_weight=weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=weight)),
        "tss": float(tpr[best] - fpr[best]),
        "threshold_youden": float(thresholds[best]),
    }


def fit_bundle(algorithm: str, table: pd.DataFrame, features: list[str], params: dict[str, Any]) -> dict[str, Any]:
    medians = table[features].median()
    if medians.isna().any():
        raise ValueError(f"Predictors without finite medians: {medians[medians.isna()].index.tolist()}")
    x = table[features].fillna(medians)
    y = table["label"].astype(int)
    weights = model_sample_weights(table.reset_index(drop=True))
    if algorithm == "rf":
        model = RandomForestClassifier(**params).fit(x, y, sample_weight=weights)
    elif algorithm == "xgb":
        if xgboost is None:
            raise RuntimeError("XGBoost was requested but is not installed.")
        model = xgboost.XGBClassifier(**params).fit(x, y, sample_weight=weights)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return {"algorithm": algorithm, "model": model, "medians": medians, "features": list(features)}


def predict_bundle(bundle: dict[str, Any], table: pd.DataFrame) -> np.ndarray:
    x = table[bundle["features"]].fillna(bundle["medians"])
    return bundle["model"].predict_proba(x)[:, 1]


def predict_ensemble(ensemble: list[dict[str, Any]], table: pd.DataFrame, return_std: bool = False):
    matrix = np.vstack([predict_bundle(bundle, table) for bundle in ensemble])
    mean = matrix.mean(axis=0)
    return (mean, matrix.std(axis=0)) if return_std else mean


def daily_block_bootstrap(presences: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Resample observed days within each bear while retaining all fixes from selected days."""
    rng = np.random.default_rng(seed)
    parts = []
    for bear, bear_rows in presences.groupby("bear_name"):
        days = bear_rows["daily_group"].dropna().unique()
        sampled_days = rng.choice(days, len(days), replace=True)
        for occurrence, day in enumerate(sampled_days):
            selected = bear_rows[bear_rows["daily_group"] == day].copy()
            selected["daily_group"] = selected["daily_group"].astype(str) + f"_bootstrap_{occurrence}"
            parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def spatial_block_ids(table: pd.DataFrame, n_axis: int) -> pd.Series:
    """Create geographic blocks whose boundaries are defined by presence locations.

    Background cells usually outnumber presences by orders of magnitude. Using all rows
    to derive quantiles can place every presence in one block, leaving no valid spatial
    validation fold. Presence-derived boundaries keep the blocks relevant to transfer
    across the observed distribution while assigning background to the same geography.
    """
    presences = table.loc[table["label"].astype(int) == 1]
    if presences.empty:
        raise ValueError("Spatial blocking requires at least one presence row.")
    quantiles = np.arange(1, n_axis) / n_axis
    x_edges = np.unique(np.quantile(presences["x_3035"], quantiles))
    y_edges = np.unique(np.quantile(presences["y_3035"], quantiles))
    x_bins = pd.Series(np.digitize(table["x_3035"], x_edges), index=table.index, dtype=int)
    y_bins = pd.Series(np.digitize(table["y_3035"], y_edges), index=table.index, dtype=int)
    return x_bins * n_axis + y_bins


def grouped_day_cv_score(algorithm: str, params: dict[str, Any], table: pd.DataFrame, features: list[str]) -> float:
    """Fallback inner validation that keeps fixes from the same observed day together."""
    positive = table["label"].astype(int).eq(1)
    groups = np.where(
        positive,
        table["daily_group"].fillna(table["point_id"]).astype(str),
        "background_" + table["point_id"].astype(str),
    )
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in splitter.split(table, table["label"], groups):
        train = table.iloc[train_index].reset_index(drop=True)
        test = table.iloc[test_index].reset_index(drop=True)
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        bundle = fit_bundle(algorithm, train, features, params)
        scores.append(roc_auc_score(test["label"], predict_bundle(bundle, test)))
    return float(np.mean(scores)) if scores else 0.5


def spatial_cv_score(algorithm: str, params: dict[str, Any], table: pd.DataFrame, features: list[str], n_axis: int) -> float:
    blocks = spatial_block_ids(table, n_axis)
    scores = []
    for block in sorted(blocks.unique()):
        train = table.loc[blocks != block].reset_index(drop=True)
        test = table.loc[blocks == block].reset_index(drop=True)
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        bundle = fit_bundle(algorithm, train, features, params)
        scores.append(roc_auc_score(test["label"], predict_bundle(bundle, test)))
    return float(np.mean(scores)) if scores else grouped_day_cv_score(algorithm, params, table, features)


def tune_algorithm(
    algorithm: str,
    table: pd.DataFrame,
    features: list[str],
    label: str,
    config: WorkflowConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not config.run_tuning:
        return dict(DEFAULT_RF_PARAMS if algorithm == "rf" else DEFAULT_XGB_PARAMS)
    if algorithm == "rf":
        candidates = ParameterSampler(RF_PARAM_DIST, n_iter=config.n_rf_tuning_iter, random_state=config.random_seed)
        best_score, best = -np.inf, None
        for index, candidate in enumerate(candidates, start=1):
            params = {**candidate, "n_jobs": -1, "random_state": config.random_seed}
            score = spatial_cv_score("rf", params, table, features, config.inner_blocks_axis)
            logger.info("%s RF candidate %d/%d AUC=%.3f", label, index, config.n_rf_tuning_iter, score)
            if score > best_score:
                best_score, best = score, params
        return best or dict(DEFAULT_RF_PARAMS)
    if xgboost is None:
        raise RuntimeError("XGBoost is unavailable.")
    if optuna is None:
        return dict(DEFAULT_XGB_PARAMS)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5, log=True),
            "tree_method": "hist",
            "eval_metric": "auc",
            "random_state": config.random_seed,
            "n_jobs": -1,
        }
        return spatial_cv_score("xgb", params, table, features, config.inner_blocks_axis)

    study = optuna.create_study(direction="maximize", study_name=label)
    study.optimize(objective, n_trials=config.n_optuna_trials)
    best = dict(study.best_params)
    best.update({"tree_method": "hist", "eval_metric": "auc", "random_state": config.random_seed, "n_jobs": -1})
    logger.info("%s XGBoost best spatial AUC=%.3f", label, study.best_value)
    return best


def _evaluation_weights(test: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(test), dtype=float)
    positive = test["label"].to_numpy() == 1
    result[positive] = 0.5 / max(positive.sum(), 1)
    result[~positive] = 0.5 / max((~positive).sum(), 1)
    return result


def evaluate_base_models(
    model_data: dict[str, dict[str, Any]],
    features: list[str],
    config: WorkflowConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    records = []
    predictions = []
    fold_models: dict[tuple[str, str, str], dict[str, Any]] = {}
    algorithms = ["rf"] + (["xgb"] if config.run_xgboost and xgboost is not None else [])
    for cub_class in CUB_CLASSES:
        data = model_data[cub_class]
        presences = data["presences"]
        for training_domain, train_pool in (("full_aoi", "full_train"), ("local_domain", "local_train")):
            background = data[train_pool]
            for algorithm in algorithms:
                for fold_index, bear in enumerate(sorted(presences["bear_name"].unique())):
                    train_presence = presences[presences["bear_name"] != bear].reset_index(drop=True)
                    test_presence = presences[presences["bear_name"] == bear].reset_index(drop=True)
                    if bear in set(train_presence["bear_name"]):
                        raise AssertionError("Held-out bear leaked into training presences.")
                    outer_train = pd.concat([train_presence, background], ignore_index=True)
                    params = (
                        tune_algorithm(
                            algorithm,
                            outer_train,
                            features,
                            f"{training_domain}_{cub_class}_{algorithm}_{bear}",
                            config,
                            logger,
                        )
                        if config.strict_nested_tuning
                        else dict(DEFAULT_RF_PARAMS if algorithm == "rf" else DEFAULT_XGB_PARAMS)
                    )
                    ensemble = []
                    for replicate in range(config.n_evaluation_replicates):
                        boot = daily_block_bootstrap(train_presence, config.random_seed + fold_index * 1_000 + replicate)
                        ensemble.append(fit_bundle(algorithm, pd.concat([boot, background], ignore_index=True), features, params))
                    fold_models[(training_domain, cub_class, algorithm, bear)] = {
                        "ensemble": ensemble,
                        "params": params,
                    }
                    for scenario, test_pool in (("local_test", "local_test"), ("full_test", "full_test")):
                        test = pd.concat([test_presence, data[test_pool]], ignore_index=True)
                        scores = predict_ensemble(ensemble, test)
                        metrics = evaluate_predictions(test["label"], scores, _evaluation_weights(test))
                        records.append(
                            {
                                "training_domain": training_domain,
                                "cub_class": cub_class,
                                "candidate": algorithm,
                                "algorithm": algorithm,
                                "fold": bear,
                                "evaluation_scenario": scenario,
                                "n_test_presence": len(test_presence),
                                "n_test_background": len(data[test_pool]),
                                **metrics,
                            }
                        )
                        part = test[["point_id", "source", "label", "bear_name", "event_season"]].copy()
                        part["score"] = scores
                        part["training_domain"] = training_domain
                        part["cub_class"] = cub_class
                        part["candidate"] = algorithm
                        part["fold"] = bear
                        part["evaluation_scenario"] = scenario
                        predictions.append(part)
                        logger.info(
                            "%s %s %s held-out %s on %s AUC=%.3f",
                            training_domain,
                            cub_class,
                            algorithm,
                            bear,
                            scenario,
                            metrics["auc_roc"],
                        )
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True), fold_models


def add_blend_candidates(
    base_results: pd.DataFrame,
    base_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "xgb" not in set(base_predictions["candidate"]):
        return base_results, base_predictions
    blend_results = []
    blend_predictions = []
    keys = ["training_domain", "cub_class", "fold", "evaluation_scenario", "point_id", "source", "label", "bear_name", "event_season"]
    rf = base_predictions[base_predictions["candidate"] == "rf"].drop(columns="candidate").rename(columns={"score": "rf_score"})
    xgb = base_predictions[base_predictions["candidate"] == "xgb"].drop(columns="candidate").rename(columns={"score": "xgb_score"})
    merged = rf.merge(xgb, on=keys, validate="one_to_one")
    for rf_weight in (0.25, 0.5, 0.75):
        name = f"blend_rf{int(rf_weight * 100)}_xgb{int((1 - rf_weight) * 100)}"
        candidate = merged[keys].copy()
        candidate["score"] = rf_weight * merged["rf_score"] + (1 - rf_weight) * merged["xgb_score"]
        candidate["candidate"] = name
        blend_predictions.append(candidate)
        for group_key, group in candidate.groupby(["training_domain", "cub_class", "fold", "evaluation_scenario"]):
            metrics = evaluate_predictions(group["label"], group["score"], _evaluation_weights(group))
            blend_results.append(
                {
                    "training_domain": group_key[0],
                    "cub_class": group_key[1],
                    "candidate": name,
                    "algorithm": "blend",
                    "fold": group_key[2],
                    "evaluation_scenario": group_key[3],
                    "n_test_presence": int((group["label"] == 1).sum()),
                    "n_test_background": int((group["label"] == 0).sum()),
                    **metrics,
                }
            )
    return (
        pd.concat([base_results, pd.DataFrame(blend_results)], ignore_index=True),
        pd.concat([base_predictions, *blend_predictions], ignore_index=True),
    )


def summarize_loio(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc_roc", "auc_pr", "boyce", "log_loss", "brier", "tss"]
    summary = results.groupby(
        ["training_domain", "cub_class", "candidate", "algorithm", "evaluation_scenario"], as_index=False
    )[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(filter(None, column)).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def select_winners(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cub_class, class_summary in summary.groupby("cub_class"):
        local = class_summary[class_summary["evaluation_scenario"] == "local_test"].copy()
        full = class_summary[class_summary["evaluation_scenario"] == "full_test"].copy()
        keys = ["training_domain", "cub_class", "candidate", "algorithm"]
        comparison = local.merge(full, on=keys, suffixes=("_local", "_full"), validate="one_to_one")
        comparison["robust_floor"] = comparison[["auc_roc_mean_local", "auc_roc_mean_full", "auc_roc_min_local"]].min(axis=1)
        comparison["mean_auc"] = comparison[["auc_roc_mean_local", "auc_roc_mean_full"]].mean(axis=1)
        comparison["mean_boyce"] = comparison[["boyce_mean_local", "boyce_mean_full"]].mean(axis=1)
        comparison["mean_brier"] = comparison[["brier_mean_local", "brier_mean_full"]].mean(axis=1)
        comparison["management_ready"] = comparison["boyce_mean_local"] >= 0
        eligible = comparison[comparison["management_ready"]]
        ranked = (eligible if not eligible.empty else comparison).sort_values(
            ["robust_floor", "mean_auc", "mean_boyce", "mean_brier"],
            ascending=[False, False, False, True],
        )
        winner = ranked.iloc[0].to_dict()
        winner["reliability_status"] = "management_ready" if winner["management_ready"] else "exploratory"
        rows.append(winner)
    return pd.DataFrame(rows)


def candidate_components(candidate: str) -> tuple[float, float]:
    if candidate == "rf":
        return 1.0, 0.0
    if candidate == "xgb":
        return 0.0, 1.0
    parts = candidate.replace("blend_rf", "").split("_xgb")
    return int(parts[0]) / 100, int(parts[1]) / 100


def predict_candidate(
    candidate: str,
    ensembles: dict[str, list[dict[str, Any]]],
    table: pd.DataFrame,
    return_components: bool = False,
):
    rf_weight, xgb_weight = candidate_components(candidate)
    rf_matrix = np.vstack([predict_bundle(bundle, table) for bundle in ensembles.get("rf", [])]) if rf_weight else None
    xgb_matrix = np.vstack([predict_bundle(bundle, table) for bundle in ensembles.get("xgb", [])]) if xgb_weight else None
    matrices = []
    if rf_matrix is not None:
        matrices.append(rf_weight * rf_matrix)
    if xgb_matrix is not None:
        matrices.append(xgb_weight * xgb_matrix)
    count = max(matrix.shape[0] for matrix in matrices)
    expanded = [matrix if matrix.shape[0] == count else np.repeat(matrix.mean(axis=0, keepdims=True), count, axis=0) for matrix in matrices]
    combined = np.sum(expanded, axis=0)
    if return_components:
        return combined.mean(axis=0), combined.std(axis=0), rf_matrix, xgb_matrix
    return combined.mean(axis=0)


def fit_final_models(
    model_data: dict[str, dict[str, Any]],
    features: list[str],
    winners: pd.DataFrame,
    config: WorkflowConfig,
    paths: RunPaths,
    logger: logging.Logger,
) -> tuple[dict[str, dict[str, dict[str, list[dict[str, Any]]]]], dict[str, Any]]:
    models: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    parameters: dict[str, Any] = {}
    for cub_class in CUB_CLASSES:
        models[cub_class] = {}
        parameters[cub_class] = {}
        data = model_data[cub_class]
        for domain, pool_name in (("full_aoi", "full_train"), ("local_domain", "local_train")):
            table = pd.concat([data["presences"], data[pool_name]], ignore_index=True)
            models[cub_class][domain] = {}
            parameters[cub_class][domain] = {}
            for algorithm in ["rf"] + (["xgb"] if config.run_xgboost and xgboost is not None else []):
                params = tune_algorithm(algorithm, table, features, f"final_{cub_class}_{domain}_{algorithm}", config, logger)
                parameters[cub_class][domain][algorithm] = params
                ensemble = []
                for replicate in range(config.n_final_bootstrap_replicates):
                    boot = daily_block_bootstrap(data["presences"], config.random_seed + 50_000 + replicate)
                    ensemble.append(fit_bundle(algorithm, pd.concat([boot, data[pool_name]], ignore_index=True), features, params))
                models[cub_class][domain][algorithm] = ensemble
                destination = paths.models / f"ensemble_{cub_class}_{domain}_{algorithm}.pkl"
                if joblib is not None:
                    joblib.dump(ensemble, destination)
                else:
                    destination.write_bytes(pickle.dumps(ensemble))
    (paths.models / "tuned_params.json").write_text(json.dumps(parameters, indent=2, default=str))
    (paths.models / "winner_selection.json").write_text(winners.to_json(orient="records", indent=2))
    return models, parameters


def fit_score_calibrator(oof: pd.DataFrame) -> dict[str, Any]:
    y = oof["label"].astype(int).to_numpy()
    scores = oof["score"].astype(float).to_numpy()
    weights = _evaluation_weights(oof)
    if len(np.unique(scores)) >= 10 and y.sum() >= 20:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(scores, y, sample_weight=weights)
        calibrated = isotonic.predict(scores)
        if np.unique(np.round(calibrated, 8)).size >= 3:
            return {"method": "isotonic", "model": isotonic}
    logistic = LogisticRegression(random_state=42)
    logistic.fit(scores.reshape(-1, 1), y, sample_weight=weights)
    return {"method": "logistic", "model": logistic}


def calibrate_scores(calibrator: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if calibrator["method"] == "isotonic":
        return np.asarray(calibrator["model"].predict(values), dtype=float)
    return calibrator["model"].predict_proba(values.reshape(-1, 1))[:, 1]


def paper_style_random_validation(
    model_data: dict[str, dict[str, Any]],
    features: list[str],
    winners: pd.DataFrame,
    parameters: dict[str, Any],
    config: WorkflowConfig,
) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(config.random_seed + 70_000)
    for winner in winners.itertuples(index=False):
        cub_class = winner.cub_class
        domain = winner.training_domain
        candidate = winner.candidate
        pool_name = "local_train" if domain == "local_domain" else "full_train"
        presences = model_data[cub_class]["presences"]
        background = model_data[cub_class][pool_name]
        for repeat in range(config.n_paper_validation_repeats):
            presence_test_ids = rng.choice(len(presences), max(1, math.ceil(0.25 * len(presences))), replace=False)
            background_test_ids = rng.choice(len(background), max(1, math.ceil(0.25 * len(background))), replace=False)
            presence_test_mask = np.zeros(len(presences), dtype=bool)
            background_test_mask = np.zeros(len(background), dtype=bool)
            presence_test_mask[presence_test_ids] = True
            background_test_mask[background_test_ids] = True
            train = pd.concat([presences.loc[~presence_test_mask], background.loc[~background_test_mask]], ignore_index=True)
            test = pd.concat([presences.loc[presence_test_mask], background.loc[background_test_mask]], ignore_index=True)
            ensembles = {}
            rf_weight, xgb_weight = candidate_components(candidate)
            if rf_weight:
                ensembles["rf"] = [fit_bundle("rf", train, features, parameters[cub_class][domain]["rf"])]
            if xgb_weight:
                ensembles["xgb"] = [fit_bundle("xgb", train, features, parameters[cub_class][domain]["xgb"])]
            metrics = evaluate_predictions(test["label"], predict_candidate(candidate, ensembles, test), _evaluation_weights(test))
            records.append(
                {
                    "cub_class": cub_class,
                    "training_domain": domain,
                    "candidate": candidate,
                    "repeat": repeat + 1,
                    "validation_design": "paper_style_random_75_25",
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def prediction_sources(inventory: pd.DataFrame, features: list[str], season: str) -> list[dict[str, Any]]:
    sources = []
    for feature in features:
        source_name = "topography_aspect" if feature.startswith("topography_aspect_") else feature
        rows = inventory[inventory["model_variable"] == source_name]
        static = rows[rows["temporal_role"].isin(["static", "annual"])]
        if not static.empty:
            sources.append({"feature": feature, "kind": "path", "path": Path(static.iloc[0]["path"]), "transform": feature})
            continue
        seasonal = rows[(rows["temporal_role"] == "seasonal") & (rows["temporal_dimension"].astype(str) == season)]
        if not seasonal.empty:
            sources.append({"feature": feature, "kind": "path", "path": Path(seasonal.iloc[0]["path"]), "transform": None})
            continue
        if source_name == "climate_snow_cover" and season in {"summer", "autumn"}:
            sources.append({"feature": feature, "kind": "constant", "value": 0.0, "transform": None})
            continue
        raise ValueError(f"No {season} raster source for {feature}")
    return sources


def _window_feature_frame(
    sources: list[dict[str, Any]],
    open_sources: dict[str, rasterio.io.DatasetReader],
    window: Window,
    features: list[str],
) -> pd.DataFrame:
    columns = []
    for source in sources:
        if source["kind"] == "constant":
            array = np.full((int(window.height), int(window.width)), source["value"], dtype=float)
        else:
            dataset = open_sources[str(source["path"])]
            array = dataset.read(1, window=window).astype(float)
            if dataset.nodata is not None:
                array[np.isclose(array, dataset.nodata)] = np.nan
        if source["transform"] == "topography_aspect_sin":
            array = np.sin(np.deg2rad(array))
        elif source["transform"] == "topography_aspect_cos":
            array = np.cos(np.deg2rad(array))
        columns.append(array.reshape(-1))
    return pd.DataFrame(np.column_stack(columns), columns=features)


def _write_array_raster(path: Path, array: np.ndarray, profile: dict[str, Any], nodata: float = -9999.0):
    output = np.asarray(array)
    result_profile = profile.copy()
    result_profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **result_profile) as dst:
        dst.write(output.astype("float32"), 1)


def normalize_uncertainty_components(component_paths: list[Path], output_path: Path):
    arrays = []
    profile = None
    for path in component_paths:
        with rasterio.open(path) as src:
            array = src.read(1).astype(float)
            valid = np.isfinite(array) & ~np.isclose(array, src.nodata)
            scaled = np.full(array.shape, np.nan)
            if valid.any():
                scale = np.nanpercentile(array[valid], 95)
                scaled[valid] = np.clip(array[valid] / max(scale, 1e-9), 0, 1)
            arrays.append(scaled)
            profile = src.profile.copy()
    combined = np.nanmean(np.stack(arrays), axis=0)
    combined[~np.isfinite(combined)] = -9999.0
    _write_array_raster(output_path, combined, profile)


def predict_official_products(
    cub_class: str,
    winner: pd.Series,
    models: dict[str, dict[str, list[dict[str, Any]]]],
    fold_models: dict[tuple[str, str, str, str], dict[str, Any]],
    model_data: dict[str, Any],
    features: list[str],
    inventory: pd.DataFrame,
    common_mask: np.ndarray,
    grid_path: Path,
    calibrator: dict[str, Any],
    config: WorkflowConfig,
    paths: RunPaths,
) -> dict[str, Path]:
    domain = winner["training_domain"]
    candidate = winner["candidate"]
    season = config.prediction_season_by_class[cub_class]
    sources = prediction_sources(inventory, features, season)
    open_sources = {str(source["path"]): rasterio.open(source["path"]) for source in sources if source["kind"] == "path"}
    with rasterio.open(grid_path) as grid:
        profile = grid.profile.copy()
        height, width = grid.shape
        rows_per_batch = max(1, config.batch_size // width)
    train_pool = "local_train" if domain == "local_domain" else "full_train"
    train = pd.concat([model_data["presences"], model_data[train_pool]], ignore_index=True)
    minima = train[features].min().to_numpy(float)
    maxima = train[features].max().to_numpy(float)
    spans = np.maximum(maxima - minima, 1e-9)
    bears = sorted(model_data["presences"]["bear_name"].unique())
    outputs = {
        "raw": paths.maps / f"diagnostic_raw_suitability_{cub_class}.tif",
        "official": paths.maps / f"official_suitability_{cub_class}.tif",
        "bootstrap": paths.maps / f"uncertainty_bootstrap_{cub_class}.tif",
        "individual": paths.maps / f"uncertainty_individual_{cub_class}.tif",
        "algorithm": paths.maps / f"uncertainty_algorithm_{cub_class}.tif",
        "domain": paths.maps / f"uncertainty_domain_{cub_class}.tif",
        "mess": paths.maps / f"environmental_mess_{cub_class}.tif",
        "outside": paths.maps / f"outside_range_predictor_count_{cub_class}.tif",
    }
    result_profile = profile.copy()
    result_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    writers = {key: rasterio.open(path, "w", **result_profile) for key, path in outputs.items()}
    try:
        for row_start in range(0, height, rows_per_batch):
            n_rows = min(rows_per_batch, height - row_start)
            window = Window(0, row_start, width, n_rows)
            frame = _window_feature_frame(sources, open_sources, window, features)
            valid = common_mask[row_start : row_start + n_rows].reshape(-1) & frame.notna().all(axis=1).to_numpy()
            values = {key: np.full(len(frame), -9999.0, dtype="float32") for key in outputs}
            if valid.any():
                valid_frame = frame.loc[valid]
                mean, bootstrap_std, rf_matrix, xgb_matrix = predict_candidate(
                    candidate,
                    models[domain],
                    valid_frame,
                    return_components=True,
                )
                calibrated = calibrate_scores(calibrator, mean)
                data = valid_frame.to_numpy(float)
                below = (minima - data) / spans
                above = (data - maxima) / spans
                outside = ((data < minima) | (data > maxima)).sum(axis=1)
                signed = np.where(data < minima, -below, np.where(data > maxima, -above, np.minimum((data - minima) / spans, (maxima - data) / spans)))
                mess = np.min(signed, axis=1)

                fold_predictions = []
                for bear in bears:
                    component_ensembles = {}
                    rf_weight, xgb_weight = candidate_components(candidate)
                    if rf_weight:
                        component_ensembles["rf"] = fold_models[(domain, cub_class, "rf", bear)]["ensemble"]
                    if xgb_weight:
                        component_ensembles["xgb"] = fold_models[(domain, cub_class, "xgb", bear)]["ensemble"]
                    fold_predictions.append(predict_candidate(candidate, component_ensembles, valid_frame))
                individual_std = np.std(np.vstack(fold_predictions), axis=0)
                algorithm_disagreement = (
                    np.abs(rf_matrix.mean(axis=0) - xgb_matrix.mean(axis=0))
                    if rf_matrix is not None and xgb_matrix is not None
                    else np.zeros(len(valid_frame))
                )
                other_domain = "local_domain" if domain == "full_aoi" else "full_aoi"
                domain_disagreement = np.abs(mean - predict_candidate(candidate, models[other_domain], valid_frame))
                official = calibrated.copy()
                if domain == "local_domain":
                    official[outside > 0] = -9999.0
                values["raw"][valid] = mean
                values["official"][valid] = official
                values["bootstrap"][valid] = bootstrap_std
                values["individual"][valid] = individual_std
                values["algorithm"][valid] = algorithm_disagreement
                values["domain"][valid] = domain_disagreement
                values["mess"][valid] = mess
                values["outside"][valid] = outside
            for key, writer in writers.items():
                writer.write(values[key].reshape(n_rows, width), 1, window=window)
    finally:
        for writer in writers.values():
            writer.close()
        for source in open_sources.values():
            source.close()
    outputs["relative"] = paths.maps / f"relative_uncertainty_{cub_class}.tif"
    normalize_uncertainty_components(
        [outputs["bootstrap"], outputs["individual"], outputs["algorithm"], outputs["domain"]],
        outputs["relative"],
    )
    return outputs


def map_mean_above(path: Path, threshold: float) -> float:
    total = 0.0
    count = 0
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window).astype(float)
            valid = np.isfinite(data) & ~np.isclose(data, src.nodata) & (data >= threshold)
            total += float(data[valid].sum())
            count += int(valid.sum())
    return total / count if count else float(threshold)


def cartobio_thresholds(presence_scores: Iterable[float], map_path: Path) -> dict[str, float]:
    scores = np.sort(np.asarray(list(presence_scores), dtype=float))
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("No finite presence scores for CARTOBIO thresholds.")
    worst_count = max(1, math.ceil(0.10 * len(scores)))
    adequate = float(scores[:worst_count].mean())
    good = map_mean_above(map_path, adequate)
    optimal = map_mean_above(map_path, good)
    return {"adequate": adequate, "good": good, "optimal": optimal}


def remove_small_patches(mask: np.ndarray, pixel_area_km2: float, minimum_km2: float = 0.5) -> np.ndarray:
    minimum_pixels = max(1, math.ceil(minimum_km2 / pixel_area_km2))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def classify_zones(suitability_path: Path, thresholds: dict[str, float], output_path: Path) -> pd.DataFrame:
    with rasterio.open(suitability_path) as src:
        suitability = src.read(1).astype(float)
        valid = np.isfinite(suitability) & ~np.isclose(suitability, src.nodata)
        pixel_area = abs(src.transform.a * src.transform.e) / 1_000_000
        profile = src.profile.copy()
    masks = {
        name: remove_small_patches(valid & (suitability >= threshold), pixel_area)
        for name, threshold in thresholds.items()
    }
    zones = np.full(suitability.shape, 255, dtype="uint8")
    zones[valid] = 0
    for value, name in enumerate(("adequate", "good", "optimal"), start=1):
        zones[masks[name]] = value
    profile.update(dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(zones, 1)
    records = []
    for value, name in enumerate(("adequate", "good", "optimal"), start=1):
        records.append({"zone": name, "area_type": "exclusive", "area_km2": float((zones == value).sum() * pixel_area)})
        records.append({"zone": name, "area_type": "cumulative", "area_km2": float(((zones != 255) & (zones >= value)).sum() * pixel_area)})
    return pd.DataFrame(records)


def plot_raster_4326(
    path: Path,
    output_path: Path,
    title: str,
    bounds_4326: dict[str, float],
    categorical: bool = False,
):
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.nearest if categorical else Resampling.bilinear) as vrt:
            scale = max(vrt.width / 1_600, vrt.height / 900, 1)
            data = vrt.read(
                1,
                out_shape=(max(1, int(vrt.height / scale)), max(1, int(vrt.width / scale))),
                masked=True,
            )
            extent = (vrt.bounds.left, vrt.bounds.right, vrt.bounds.bottom, vrt.bounds.top)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    image = ax.imshow(data, extent=extent, origin="upper", cmap="plasma" if categorical else "viridis")
    ax.set(title=title, xlabel="Longitude (EPSG:4326)", ylabel="Latitude (EPSG:4326)")
    ax.set_xlim(bounds_4326["xmin"], bounds_4326["xmax"])
    ax.set_ylim(bounds_4326["ymin"], bounds_4326["ymax"])
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def oof_permutation_importance(
    cub_class: str,
    winner: pd.Series,
    model_data: dict[str, Any],
    features: list[str],
    fold_models: dict[tuple[str, str, str, str], dict[str, Any]],
    config: WorkflowConfig,
) -> pd.DataFrame:
    domain, candidate = winner["training_domain"], winner["candidate"]
    rng = np.random.default_rng(config.random_seed + 80_000)
    records = []
    repeats = 1 if config.smoke_mode else 3
    for bear in sorted(model_data["presences"]["bear_name"].unique()):
        test_presence = model_data["presences"][model_data["presences"]["bear_name"] == bear]
        test = pd.concat([test_presence, model_data["local_test"]], ignore_index=True)
        components = {}
        rf_weight, xgb_weight = candidate_components(candidate)
        if rf_weight:
            components["rf"] = fold_models[(domain, cub_class, "rf", bear)]["ensemble"]
        if xgb_weight:
            components["xgb"] = fold_models[(domain, cub_class, "xgb", bear)]["ensemble"]
        baseline = roc_auc_score(test["label"], predict_candidate(candidate, components, test))
        for feature in features:
            losses = []
            for _ in range(repeats):
                shuffled = test.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                losses.append(baseline - roc_auc_score(test["label"], predict_candidate(candidate, components, shuffled)))
            records.append({"cub_class": cub_class, "fold": bear, "feature": feature, "auc_loss": float(np.mean(losses))})
    result = pd.DataFrame(records)
    return result.groupby(["cub_class", "feature"], as_index=False)["auc_loss"].agg(["mean", "std"]).reset_index()


def plot_readable_shap(
    algorithm: str,
    models: dict[str, dict[str, list[dict[str, Any]]]],
    training_domain: str,
    model_data: dict[str, Any],
    features: list[str],
    output_path: Path,
    seed: int,
):
    if shap is None or algorithm not in models[training_domain] or not models[training_domain][algorithm]:
        return
    bundle = models[training_domain][algorithm][0]
    sample = model_data["presences"].sample(min(2_000, len(model_data["presences"])), random_state=seed)
    x = sample[features].fillna(bundle["medians"])
    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]
    elif np.ndim(values) == 3:
        values = values[:, :, -1]
    display_x = x.rename(columns=SHORT_FEATURE_LABELS)
    shap.summary_plot(values, display_x, max_display=15, plot_size=(11, 8), show=False)
    plt.gcf().subplots_adjust(left=0.34, right=0.92, bottom=0.12)
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def independent_obs_audit(
    obs: pd.DataFrame,
    model_data: dict[str, dict[str, Any]],
    features: list[str],
    winners: pd.DataFrame,
    models: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for winner in winners.itertuples(index=False):
        subset = obs[obs["cub_class"] == winner.cub_class].copy()
        if subset.empty:
            continue
        subset["point_id"] = subset["id_obs"].astype(str)
        subset["label"] = 1
        subset["source"] = "obs"
        subset_features = extract_features(subset, inventory)
        obs_table = pd.concat([subset.reset_index(drop=True), subset_features.reset_index(drop=True)], axis=1)
        obs_table = obs_table.dropna(subset=features)
        background = model_data[winner.cub_class]["full_test"].sample(min(len(obs_table), len(model_data[winner.cub_class]["full_test"])), random_state=42)
        test = pd.concat([obs_table, background], ignore_index=True)
        scores = predict_candidate(winner.candidate, models[winner.cub_class][winner.training_domain], test)
        metrics = evaluate_predictions(test["label"], scores, _evaluation_weights(test))
        positive = scores[test["label"].to_numpy() == 1]
        negative = scores[test["label"].to_numpy() == 0]
        records.append(
            {
                "cub_class": winner.cub_class,
                "training_domain": winner.training_domain,
                "candidate": winner.candidate,
                "n_obs_presence": len(positive),
                "n_background": len(negative),
                "mean_obs_suitability": float(np.mean(positive)),
                "mean_background_suitability": float(np.mean(negative)),
                "mannwhitney_p_greater": float(mannwhitneyu(positive, negative, alternative="greater").pvalue),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def write_geojson(geometries: dict[str, Any], path: Path, crs: str):
    content = {
        "type": "FeatureCollection",
        "name": "ursus_arctos_modelling_domains",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [
            {"type": "Feature", "properties": {"domain": name}, "geometry": mapping(geometry)}
            for name, geometry in geometries.items()
        ],
    }
    path.write_text(json.dumps(content, indent=2))


def output_manifest(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "output_manifest.csv":
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_mb": path.stat().st_size / 1_000_000,
                    "modified_utc": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True),
                }
            )
    return pd.DataFrame(records)


def capture_run_metadata(project_root: Path, config: WorkflowConfig, paths: RunPaths, inputs: list[Path]):
    (paths.root / "parameters.json").write_text(json.dumps(asdict(config), indent=2, default=str))
    status = subprocess.run(["git", "status", "--short"], cwd=project_root, capture_output=True, text=True, check=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False)
    (paths.root / "git_state.txt").write_text(f"commit: {revision.stdout.strip()}\n\n{status.stdout}")
    input_table = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(project_root)),
                "size_mb": path.stat().st_size / 1_000_000,
                "modified_utc": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True),
            }
            for path in inputs
            if path.exists()
        ]
    )
    input_table.to_csv(paths.root / "input_files.csv", index=False)


def plot_domains_4326(
    presences: pd.DataFrame,
    pools: dict[str, pd.DataFrame],
    geometries: dict[str, Any],
    source_crs: str,
    output_path: Path,
):
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    to_lonlat = lambda x, y, z=None: transformer.transform(x, y)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, domain, pool_name in zip(axes, ("full_aoi", "local_domain"), ("full_train", "local_train")):
        geometry = transform_geometry(to_lonlat, geometries[domain])
        pieces = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        for piece in pieces:
            x, y = piece.exterior.xy
            ax.plot(x, y, color="black", linewidth=1)
        background = pools[pool_name]
        lon, lat = transformer.transform(background["x_3035"].to_numpy(), background["y_3035"].to_numpy())
        ax.scatter(lon, lat, s=1, alpha=0.15, color="#4c78a8", label="Training background")
        presence_lon, presence_lat = transformer.transform(presences["x_3035"].to_numpy(), presences["y_3035"].to_numpy())
        ax.scatter(presence_lon, presence_lat, s=4, alpha=0.55, color="#d64b3c", label="GPS presence")
        ax.set(title=domain.replace("_", " "), xlabel="Longitude (EPSG:4326)", ylabel="Latitude (EPSG:4326)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


class HabitatWorkflow:
    """Stateful workflow used by the notebook's documented analytical stages."""

    def __init__(self, config: WorkflowConfig, project_root: Path | None = None):
        self.config = config.effective()
        self.project_root = project_root or find_project_root(Path.cwd())
        os.chdir(self.project_root)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/pirineus_raster_matplotlib")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        self.paths = RunPaths.create((self.project_root / self.config.output_dir).resolve())
        self.logger = setup_logger(self.paths.logs / "sdm_bears.log")
        self._resolve_inputs()
        self.logger.info("Project root: %s", self.project_root)
        self.logger.info("Run root: %s", self.paths.root)

    def _resolve_inputs(self):
        for name in (
            "raster_dir", "gps_path", "obs_path", "individuals_path", "grid_path",
            "aoi_config_path", "run_config_path", "maxent_reference_dir",
        ):
            value = Path(getattr(self.config, name))
            setattr(self, name, value if value.is_absolute() else self.project_root / value)
        required = [
            self.raster_dir, self.gps_path, self.obs_path, self.individuals_path,
            self.grid_path, self.aoi_config_path, self.run_config_path,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required inputs: {missing}")

    def inspect_and_treat_data(self) -> dict[str, pd.DataFrame]:
        """Inspect environmental inputs and recover/treat biological records."""
        self.aoi_config = yaml.safe_load(self.aoi_config_path.read_text())
        self.target_crs = self.aoi_config["crs"]
        self.bounds_4326 = self.aoi_config["bounds_epsg4326"]
        self.inventory = load_raster_inventory(self.raster_dir)
        alignment = check_alignment(self.inventory, self.grid_path)
        if not alignment[["same_shape", "same_transform", "same_bounds", "same_crs"]].all(axis=None):
            raise ValueError("Environmental rasters are not aligned with the official grid.")
        quality = profile_environment(self.inventory)
        self.common_mask = build_common_mask(self.inventory, self.grid_path, self.paths.intermediate / "common_predictor_mask.tif")
        gps, obs, gps_report, obs_report, failed_obs = clean_biological_data(
            self.gps_path,
            self.obs_path,
            self.individuals_path,
            self.target_crs,
            self.bounds_4326,
        )
        self.gps = assign_grid_cells(gps, self.grid_path)
        self.obs = assign_grid_cells(obs, self.grid_path)
        self.presences, dedup_report, thinning_report = treat_presences(self.gps, self.config)
        self.inventory.to_csv(self.paths.tables / "environmental_raster_inventory.csv", index=False)
        self.inventory[self.inventory["temporal_role"] == "seasonal"].to_csv(
            self.paths.tables / "environmental_temporal_variables.csv", index=False
        )
        alignment.to_csv(self.paths.tables / "raster_alignment_check.csv", index=False)
        quality.to_csv(self.paths.tables / "environmental_dataset_quality_report.csv", index=False)
        gps_report.to_csv(self.paths.tables / "gps_cleaning_report.csv", index=False)
        obs_report.to_csv(self.paths.tables / "obs_cleaning_report.csv", index=False)
        failed_obs.to_csv(self.paths.tables / "obs_unparseable_dates.csv", index=False)
        dedup_report.to_csv(self.paths.tables / "gps_grid_cell_deduplication_report.csv", index=False)
        thinning_report.to_csv(self.paths.tables / "gps_spatial_thinning_report.csv", index=False)
        summary = self.presences.groupby(["cub_class", "bear_name", "event_season"], as_index=False).agg(
            n_presences=("id_obs", "size"), date_min=("datetime_utc", "min"), date_max=("datetime_utc", "max"), n_days=("daily_group", "nunique")
        )
        summary.to_csv(self.paths.tables / "individual_presence_summary.csv", index=False)
        if len(failed_obs):
            self.logger.warning("%d OBS records still have unparseable dates.", len(failed_obs))
        else:
            self.logger.info("All target OBS records have parseable dates and seasons.")
        capture_run_metadata(
            self.project_root,
            self.config,
            self.paths,
            [self.gps_path, self.obs_path, self.individuals_path, self.grid_path, self.aoi_config_path, self.run_config_path]
            + [Path(path) for path in self.inventory["path"]],
        )
        return {
            "environmental_quality": quality,
            "gps_cleaning": gps_report,
            "obs_cleaning": obs_report,
            "presence_summary": summary,
        }

    def prepare_domains_and_model_data(self) -> dict[str, pd.DataFrame]:
        """Build local/full domains, isolated background pools, and seasonal model tables."""
        self.domain_geometries, sensitivity = build_domain_geometries(
            self.presences,
            self.grid_path,
            self.target_crs,
            self.config,
        )
        selected_geometry = self.domain_geometries["local_domain"]
        selected_components = len(selected_geometry.geoms) if hasattr(selected_geometry, "geoms") else 1
        if not self.config.custom_local_domain_path and np.isclose(self.config.local_buffer_km, 25) and selected_components != 2:
            raise AssertionError(f"Expected two local-domain components at 25 km, found {selected_components}.")
        self.domain_masks = {
            name: domain_mask(geometry, self.grid_path) & self.common_mask
            for name, geometry in self.domain_geometries.items()
        }
        self.background_pools = create_background_pools(self.domain_masks, self.grid_path, self.config)
        self.model_data, self.features = build_model_data(self.presences, self.background_pools, self.inventory)
        sensitivity.to_csv(self.paths.tables / "local_domain_sensitivity.csv", index=False)
        domain_summary = pd.DataFrame(
            [
                {
                    "domain": name,
                    "area_km2": geometry.area / 1_000_000,
                    "valid_area_km2": self.domain_masks[name].sum() * 0.01,
                    "n_components": len(geometry.geoms) if hasattr(geometry, "geoms") else 1,
                }
                for name, geometry in self.domain_geometries.items()
            ]
        )
        domain_summary.to_csv(self.paths.tables / "modelling_domain_summary.csv", index=False)
        write_geojson(self.domain_geometries, self.paths.intermediate / "modelling_domains.geojson", self.target_crs)
        for name, pool in self.background_pools.items():
            pool.to_csv(self.paths.intermediate / f"background_{name}.csv", index=False)
        for cub_class, data in self.model_data.items():
            data["presences"].to_csv(self.paths.intermediate / f"presences_{cub_class}.csv", index=False)
            for pool_name in self.background_pools:
                data[pool_name].to_csv(self.paths.intermediate / f"features_{cub_class}_{pool_name}.csv", index=False)
        feature_table = pd.DataFrame(
            {
                "feature": sorted(set(self.features) | REDUCED_FEATURE_EXCLUSIONS),
                "in_reduced_set": [feature in self.features for feature in sorted(set(self.features) | REDUCED_FEATURE_EXCLUSIONS)],
            }
        )
        feature_table.to_csv(self.paths.tables / "model_feature_sets.csv", index=False)
        plot_domains_4326(
            self.presences,
            self.background_pools,
            self.domain_geometries,
            self.target_crs,
            self.paths.plots / "modelling_domains_backgrounds_epsg4326.png",
        )
        return {"domain_summary": domain_summary, "domain_sensitivity": sensitivity, "feature_set": feature_table}

    def evaluate_and_select_models(self) -> dict[str, pd.DataFrame]:
        """Run nested LOIO on shared test pools, select winners, and fit final ensembles."""
        base_results, base_predictions, self.fold_models = evaluate_base_models(
            self.model_data,
            self.features,
            self.config,
            self.logger,
        )
        self.loio_results, self.oof_predictions = add_blend_candidates(base_results, base_predictions)
        self.loio_summary = summarize_loio(self.loio_results)
        self.winners = select_winners(self.loio_summary)
        self.loio_results.to_csv(self.paths.tables / "loio_cv_results.csv", index=False)
        self.oof_predictions.to_csv(self.paths.tables / "loio_out_of_fold_predictions.csv", index=False)
        self.loio_summary.to_csv(self.paths.tables / "loio_summary.csv", index=False)
        self.winners.to_csv(self.paths.tables / "selected_final_models.csv", index=False)
        self.final_models, self.final_parameters = fit_final_models(
            self.model_data,
            self.features,
            self.winners,
            self.config,
            self.paths,
            self.logger,
        )
        paper_validation = paper_style_random_validation(
            self.model_data,
            self.features,
            self.winners,
            self.final_parameters,
            self.config,
        )
        paper_validation.to_csv(self.paths.tables / "paper_style_random_validation.csv", index=False)
        self.calibrators = {}
        calibration_records = []
        for winner in self.winners.itertuples(index=False):
            scenario = "local_test" if winner.training_domain == "local_domain" else "full_test"
            selected = self.oof_predictions[
                (self.oof_predictions["cub_class"] == winner.cub_class)
                & (self.oof_predictions["training_domain"] == winner.training_domain)
                & (self.oof_predictions["candidate"] == winner.candidate)
                & (self.oof_predictions["evaluation_scenario"] == scenario)
            ].copy()
            calibrator = fit_score_calibrator(selected)
            self.calibrators[winner.cub_class] = calibrator
            calibrated = calibrate_scores(calibrator, selected["score"].to_numpy())
            calibration_records.append(
                {
                    "cub_class": winner.cub_class,
                    "method": calibrator["method"],
                    "scenario": scenario,
                    "raw_brier": brier_score_loss(selected["label"], selected["score"]),
                    "calibrated_brier": brier_score_loss(selected["label"], calibrated),
                }
            )
            destination = self.paths.models / f"calibrator_{winner.cub_class}.pkl"
            if joblib is not None:
                joblib.dump(calibrator, destination)
            else:
                destination.write_bytes(pickle.dumps(calibrator))
        calibration = pd.DataFrame(calibration_records)
        calibration.to_csv(self.paths.tables / "calibration_summary.csv", index=False)
        return {
            "loio_summary": self.loio_summary,
            "winners": self.winners,
            "paper_validation": paper_validation,
            "calibration": calibration,
        }

    def generate_final_products(self) -> dict[str, pd.DataFrame]:
        """Generate exactly one official continuous and categorical map per cub class."""
        threshold_records = []
        area_tables = []
        importance_tables = []
        if self.config.run_map_prediction:
            for _, winner in self.winners.iterrows():
                cub_class = winner["cub_class"]
                outputs = predict_official_products(
                    cub_class,
                    winner,
                    self.final_models[cub_class],
                    self.fold_models,
                    self.model_data[cub_class],
                    self.features,
                    self.inventory,
                    self.common_mask,
                    self.grid_path,
                    self.calibrators[cub_class],
                    self.config,
                    self.paths,
                )
                domain = winner["training_domain"]
                candidate = winner["candidate"]
                presence_scores_raw = predict_candidate(candidate, self.final_models[cub_class][domain], self.model_data[cub_class]["presences"])
                scenario = "local_test" if domain == "local_domain" else "full_test"
                winning_oof = self.oof_predictions[
                    (self.oof_predictions["cub_class"] == cub_class)
                    & (self.oof_predictions["training_domain"] == domain)
                    & (self.oof_predictions["candidate"] == candidate)
                    & (self.oof_predictions["evaluation_scenario"] == scenario)
                    & (self.oof_predictions["label"] == 1)
                ]
                conservative_presence = calibrate_scores(self.calibrators[cub_class], winning_oof["score"].to_numpy())
                paper_threshold = cartobio_thresholds(presence_scores_raw, outputs["raw"])
                conservative_threshold = cartobio_thresholds(conservative_presence, outputs["official"])
                threshold_records.extend(
                    [
                        {"cub_class": cub_class, "threshold_set": "paper_compatible", **paper_threshold},
                        {"cub_class": cub_class, "threshold_set": "conservative_oof_calibrated", **conservative_threshold},
                    ]
                )
                official_zones = self.paths.maps / f"official_zones_{cub_class}.tif"
                paper_zones = self.paths.maps / f"diagnostic_paper_compatible_zones_{cub_class}.tif"
                official_area = classify_zones(outputs["official"], conservative_threshold, official_zones)
                official_area.insert(0, "threshold_set", "conservative_oof_calibrated")
                official_area.insert(0, "cub_class", cub_class)
                paper_area = classify_zones(outputs["raw"], paper_threshold, paper_zones)
                paper_area.insert(0, "threshold_set", "paper_compatible")
                paper_area.insert(0, "cub_class", cub_class)
                area_tables.extend([official_area, paper_area])
                plot_raster_4326(
                    outputs["official"],
                    self.paths.plots / f"official_suitability_{cub_class}_epsg4326.png",
                    f"Official habitat suitability: {cub_class}",
                    self.bounds_4326,
                )
                plot_raster_4326(
                    official_zones,
                    self.paths.plots / f"official_zones_{cub_class}_epsg4326.png",
                    f"Official habitat zones: {cub_class}",
                    self.bounds_4326,
                    categorical=True,
                )
                plot_raster_4326(
                    outputs["relative"],
                    self.paths.plots / f"relative_uncertainty_{cub_class}_epsg4326.png",
                    f"Relative model uncertainty: {cub_class}",
                    self.bounds_4326,
                )
                importance = oof_permutation_importance(
                    cub_class,
                    winner,
                    self.model_data[cub_class],
                    self.features,
                    self.fold_models,
                    self.config,
                )
                importance_tables.append(importance)
                if self.config.run_shap:
                    for algorithm in ("rf", "xgb"):
                        plot_readable_shap(
                            algorithm,
                            self.final_models[cub_class],
                            domain,
                            self.model_data[cub_class],
                            self.features,
                            self.paths.plots / f"shap_beeswarm_{cub_class}_{algorithm}.png",
                            self.config.random_seed,
                        )
        thresholds = pd.DataFrame(threshold_records)
        areas = pd.concat(area_tables, ignore_index=True) if area_tables else pd.DataFrame()
        importance = pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame()
        thresholds.to_csv(self.paths.tables / "zone_thresholds.csv", index=False)
        areas.to_csv(self.paths.tables / "zone_areas.csv", index=False)
        importance.to_csv(self.paths.tables / "oof_permutation_importance.csv", index=False)
        obs_audit = independent_obs_audit(
            self.obs,
            self.model_data,
            self.features,
            self.winners,
            self.final_models,
            self.inventory,
        )
        obs_audit.to_csv(self.paths.tables / "obs_validation.csv", index=False)
        official_suitability = list(self.paths.maps.glob("official_suitability_*.tif"))
        official_zones = list(self.paths.maps.glob("official_zones_*.tif"))
        if self.config.run_map_prediction and (len(official_suitability) != 2 or len(official_zones) != 2):
            raise AssertionError("Expected exactly two official suitability maps and two official zone maps.")
        manifest = output_manifest(self.paths.root)
        manifest.to_csv(self.paths.root / "output_manifest.csv", index=False)
        self.logger.info("Workflow complete with %d run-owned output files.", len(manifest))
        return {"thresholds": thresholds, "zone_areas": areas, "importance": importance, "obs_validation": obs_audit, "manifest": manifest}

    def run_all(self) -> dict[str, Any]:
        return {
            "data": self.inspect_and_treat_data(),
            "domains": self.prepare_domains_and_model_data(),
            "models": self.evaluate_and_select_models(),
            "products": self.generate_final_products(),
        }
