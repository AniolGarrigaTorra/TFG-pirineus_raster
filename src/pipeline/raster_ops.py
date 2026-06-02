from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.io.paths import get_grid_path
from src.pipeline.resampling import (
    get_resampling_enum,
    is_conservative_resampling,
)
from src.pipeline.progress import progress_log


# =============================================================================
# Grid context
# =============================================================================


@dataclass(frozen=True, slots=True)
class GridContext:
    """
    Immutable description of the target project grid.

    All output feature rasters must match this grid exactly.
    """

    path: Path
    profile: dict[str, Any]
    transform: Affine
    crs: CRS
    height: int
    width: int
    resolution_m: int
    aoi_name: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


def load_grid_context(
    project_cfg: dict,
    aoi_cfg: dict,
    resolution_m: int,
) -> GridContext:
    """
    Load target grid metadata from the project grid raster.
    """
    aoi_name = aoi_cfg["name"]

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Target grid does not exist: {grid_path}\n"
            f"Create it first with:\n"
            f"  python -m src.make_grid "
            f"--project-config configs/project.yaml "
            f"--aoi-config <aoi_config> "
            f"--resolution {resolution_m}"
        )

    with rasterio.open(grid_path) as grid:
        profile = grid.profile.copy()

        return GridContext(
            path=grid_path,
            profile=profile,
            transform=grid.transform,
            crs=grid.crs,
            height=grid.height,
            width=grid.width,
            resolution_m=int(resolution_m),
            aoi_name=aoi_name,
        )


def print_grid_context(
    grid: GridContext,
    prefix: str = "[grid]",
) -> None:
    progress_log(f"{prefix} AOI: {grid.aoi_name}")
    progress_log(f"{prefix} Path: {grid.path}")
    progress_log(f"{prefix} CRS: {grid.crs}")
    progress_log(f"{prefix} Shape: {grid.width} x {grid.height}")
    progress_log(f"{prefix} Resolution: {grid.resolution_m} m")


# =============================================================================
# Resampling
# =============================================================================


def get_resampling_method(method_name: str | None) -> Resampling:
    """
    Convert a string resampling method name to rasterio.enums.Resampling.
    """
    return get_resampling_enum(method_name)


def get_variable_resampling_method_name(
    source_cfg: dict,
    variable: str,
) -> str:
    """
    Return resampling method name for one variable.

    Supported config patterns are intentionally flexible:

    resampling:
      default: nearest
      variables:
        elev: bilinear

    or:

    variables:
      elev:
        resampling: bilinear
    """
    variable_cfg = source_cfg.get("variables", {}).get(variable, {})
    index_cfg = source_cfg.get("indices", {}).get(variable, {})

    for key in ["resampling", "default_resampling"]:
        if key in variable_cfg:
            return str(variable_cfg[key])

    for key in ["resampling", "default_resampling"]:
        if key in index_cfg:
            return str(index_cfg[key])

    resampling_cfg = source_cfg.get("resampling", {}) or {}

    for key in ["variables", "per_variable", "by_variable"]:
        per_variable = resampling_cfg.get(key, {})
        if variable in per_variable:
            return str(per_variable[variable])

    if variable in resampling_cfg:
        return str(resampling_cfg[variable])

    return str(resampling_cfg.get("default", "nearest"))


def get_variable_resampling_method(
    source_cfg: dict,
    variable: str,
) -> Resampling:
    """
    Return rasterio Resampling enum for one variable.
    """
    return get_resampling_method(
        get_variable_resampling_method_name(
            source_cfg=source_cfg,
            variable=variable,
        )
    )


# =============================================================================
# Raster reading and grid alignment
# =============================================================================


def read_raster_to_grid(
    raster_path: Path,
    grid: GridContext,
    resampling: Resampling,
    band: int = 1,
    scale_factor: float = 1.0,
    resampling_method_name: str | None = None,
) -> np.ndarray:
    """
    Read one raster band and align it to the target project grid.

    Returns a float32 array with np.nan as internal nodata.
    Scale factor is applied after reprojection.
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Input raster does not exist: {raster_path}")

    dst = np.full(
        grid.shape,
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as src:
        src_array = src.read(band).astype(np.float32)
        src_nodata = src.nodata
        source_pixel_area_m2 = _source_pixel_area_m2(src)

        if src_nodata is not None:
            src_array = np.where(src_array == src_nodata, np.nan, src_array)

        reproject(
            source=src_array,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    if is_conservative_resampling(resampling_method_name):
        target_pixel_area_m2 = abs(grid.transform.a * grid.transform.e)
        if source_pixel_area_m2 is None or source_pixel_area_m2 <= 0:
            raise ValueError(
                "conservative_sum/extensive_sum requires a projected source raster "
                "with metre-like units so source pixel area can be estimated."
            )
        dst = dst * float(target_pixel_area_m2 / source_pixel_area_m2)

    if scale_factor != 1.0:
        dst = dst * float(scale_factor)

    dst = dst.astype(np.float32)
    dst[~np.isfinite(dst)] = np.nan

    return dst


def read_category_fraction_to_grid(
    raster_path: Path,
    grid: GridContext,
    class_values: list[int | float | str],
    resampling: Resampling = Resampling.average,
    band: int = 1,
) -> np.ndarray:
    """
    Align a categorical raster to the target grid as class coverage fraction.

    The source raster is first converted to a 0/1 mask in source space, then
    resampled with average. For upscaling, each target pixel therefore stores
    the fraction of valid source pixels belonging to the requested class/group.
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Input raster does not exist: {raster_path}")

    values = np.array([float(value) for value in class_values], dtype=np.float32)
    if values.size == 0:
        raise ValueError("class_values must contain at least one category value.")

    dst = np.full(
        grid.shape,
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as src:
        src_array = src.read(band).astype(np.float32)
        src_nodata = src.nodata

        valid = np.isfinite(src_array)
        if src_nodata is not None:
            valid &= src_array != float(src_nodata)

        mask = np.full(src_array.shape, np.nan, dtype=np.float32)
        mask[valid] = np.isin(src_array[valid], values).astype(np.float32)

        reproject(
            source=mask,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    dst = dst.astype(np.float32)
    dst[~np.isfinite(dst)] = np.nan

    return dst


def _source_pixel_area_m2(src) -> float | None:
    crs = src.crs
    if crs is None or not crs.is_projected:
        return None
    return abs(float(src.transform.a) * float(src.transform.e))


def stack_rasters_to_grid(
    raster_paths: list[Path],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float = 1.0,
    band: int = 1,
    resampling_method_name: str | None = None,
) -> np.ndarray:
    """
    Read several rasters and return a stack aligned to the target grid.

    Output shape:
      (n_layers, height, width)
    """
    arrays = [
        read_raster_to_grid(
            raster_path=path,
            grid=grid,
            resampling=resampling,
            band=band,
            scale_factor=scale_factor,
            resampling_method_name=resampling_method_name,
        )
        for path in raster_paths
    ]

    if not arrays:
        raise ValueError("No raster paths provided for stack.")

    return np.stack(arrays, axis=0)


def read_raster_array_as_nan(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a single-band raster as float32 and convert nodata to np.nan.

    Returns:
      array, profile
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)

    array[~np.isfinite(array)] = np.nan

    return array, profile


# =============================================================================
# Metadata
# =============================================================================


def _json_safe(value: Any) -> Any:
    """
    Convert common non-JSON types to JSON-safe values.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return list(value)

    return value


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None

    path = Path(path)
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_safe(value)
        for key, value in metadata.items()
        if value is not None
    }


def _infer_resolution_unit(source_cfg: dict, native_resolution_m: Any = None) -> str | None:
    if native_resolution_m is not None:
        return "m"

    provider = source_cfg.get("source", {}).get("provider")
    source_resolution = str(source_cfg.get("processing", {}).get("source_resolution", ""))

    if source_resolution in {"osm", "vector", ""}:
        return None

    if provider == "worldclim":
        if source_resolution.endswith("arcs") or source_resolution.endswith("s"):
            return "arcs"
        if source_resolution.endswith("arcmin") or source_resolution.endswith("m"):
            return "arcmin"

    if source_resolution.endswith("arcs"):
        return "arcs"
    if source_resolution.endswith("arcmin"):
        return "arcmin"

    if source_resolution.endswith("m"):
        return "m"

    return None


def _infer_metadata_resolution_unit(metadata: dict[str, Any]) -> str | None:
    if metadata.get("native_resolution_m") is not None:
        return "m"

    provider = metadata.get("provider")
    source_resolution = str(metadata.get("source_resolution", ""))

    if source_resolution in {"osm", "vector", ""}:
        return None

    if provider == "worldclim":
        if source_resolution.endswith("arcs") or source_resolution.endswith("s"):
            return "arcs"
        if source_resolution.endswith("arcmin") or source_resolution.endswith("m"):
            return "arcmin"

    if source_resolution.endswith("arcs"):
        return "arcs"
    if source_resolution.endswith("arcmin"):
        return "arcmin"

    if source_resolution.endswith("m"):
        return "m"

    return None


def _source_crs(source_cfg: dict) -> Any:
    source = source_cfg.get("source", {})
    dataset = source_cfg.get("dataset", {})
    processing = source_cfg.get("processing", {})
    source_resolution = str(processing.get("source_resolution", ""))
    by_resolution = processing.get("source_crs_by_resolution", {}) or {}
    return (
        by_resolution.get(source_resolution)
        or source.get("source_crs")
        or dataset.get("source_crs")
    )


def _native_resolution_m(source_cfg: dict) -> Any:
    dataset = source_cfg.get("dataset", {})
    processing = source_cfg.get("processing", {})
    source_resolution = str(processing.get("source_resolution", ""))
    by_resolution = processing.get("native_resolution_m_by_resolution", {}) or {}
    if source_resolution in by_resolution:
        return by_resolution[source_resolution]
    return dataset.get("native_resolution_m")


def _native_resolution(source_cfg: dict) -> Any:
    dataset = source_cfg.get("dataset", {})
    processing = source_cfg.get("processing", {})
    source_resolution = processing.get("source_resolution")
    by_resolution = processing.get("native_resolution_by_resolution", {}) or {}
    if source_resolution is not None:
        return by_resolution.get(str(source_resolution), dataset.get("native_resolution"))
    return dataset.get("native_resolution")


def _bounds_dict(grid: GridContext) -> dict[str, float]:
    left, bottom, right, top = rasterio.transform.array_bounds(
        grid.height,
        grid.width,
        grid.transform,
    )

    return {
        "xmin": float(left),
        "ymin": float(bottom),
        "xmax": float(right),
        "ymax": float(top),
    }


def enrich_output_metadata(
    metadata: dict[str, Any],
    output_path: str | Path,
    grid: GridContext,
    output_dtype: str,
    nodata: float | int,
    compression: str,
) -> dict[str, Any]:
    """
    Add provider-agnostic output provenance to one raster metadata record.
    """
    enriched = dict(metadata)

    output_path = Path(output_path)
    generated_at = enriched.get("generated_at") or datetime.now(timezone.utc).isoformat()
    source_config_path = enriched.get("source_config_path")
    source_resolution = enriched.get("source_resolution")

    if source_config_path and "source_config_sha256" not in enriched:
        enriched["source_config_sha256"] = _file_sha256(source_config_path)

    if source_resolution is not None and "native_resolution" not in enriched:
        enriched["native_resolution"] = source_resolution

    if "native_resolution_unit" not in enriched:
        native_unit = _infer_metadata_resolution_unit(enriched)
        if native_unit is not None:
            enriched["native_resolution_unit"] = native_unit

    if "source_resolution_unit" not in enriched:
        source_unit = _infer_metadata_resolution_unit(enriched)
        if source_unit is not None:
            enriched["source_resolution_unit"] = source_unit

    enriched.update(
        {
            "metadata_schema_version": "0.3",
            "generated_at": generated_at,
            "output_path": str(output_path),
            "grid_path": str(grid.path),
            "grid_aoi_name": grid.aoi_name,
            "crs": str(grid.crs),
            "output_crs": str(grid.crs),
            "target_crs": str(grid.crs),
            "resolution_m": grid.resolution_m,
            "output_resolution_m": grid.resolution_m,
            "output_width": grid.width,
            "output_height": grid.height,
            "output_shape": [grid.height, grid.width],
            "output_transform": list(grid.transform),
            "output_bounds": _bounds_dict(grid),
            "nodata": nodata,
            "dtype": output_dtype,
            "compression": compression,
        }
    )

    return _clean_metadata(enriched)


def metadata_to_geotiff_tags(metadata: dict[str, Any]) -> dict[str, str]:
    """
    Convert metadata dictionary to GeoTIFF-safe string tags.

    GeoTIFF tags should be simple string key-value pairs.
    Complex values are serialized as JSON strings.
    """
    tags: dict[str, str] = {}

    for key, value in _clean_metadata(metadata).items():
        if isinstance(value, (dict, list)):
            tags[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            tags[key] = str(value)

    return tags


def write_sidecar_json(
    metadata: dict[str, Any],
    raster_path: str | Path,
) -> Path:
    """
    Write metadata next to a GeoTIFF as .json.
    """
    raster_path = Path(raster_path)
    json_path = raster_path.with_suffix(".json")

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            _clean_metadata(metadata),
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return json_path


def _source_metadata(source_cfg: dict) -> dict[str, Any]:
    source = source_cfg.get("source", {})
    dataset = source_cfg.get("dataset", {})
    processing = source_cfg.get("processing", {})
    source_config_path = source_cfg.get("_config_path")
    native_resolution_m = _native_resolution_m(source_cfg)
    native_resolution = _native_resolution(source_cfg) or processing.get("source_resolution")

    return {
        "provider": source.get("provider"),
        "product": source.get("product"),
        "product_group": source.get("product_group"),
        "source_id": source.get("id"),
        "source_config_path": source_config_path,
        "source_config_sha256": _file_sha256(source_config_path),
        "source_version": source.get("version"),
        "version": source.get("version"),
        "source_crs": _source_crs(source_cfg),
        "source_period": source.get("source_period"),
        "source_scale": source.get("source_scale"),
        "citation": source.get("citation"),
        "page_url": source.get("page_url"),
        "article_url": source.get("article_url"),
        "doi": source.get("doi"),
        "dataset_layer_structure": dataset.get("layer_structure"),
        "source_resolution": processing.get("source_resolution"),
        "source_resolution_unit": _infer_resolution_unit(source_cfg),
        "native_resolution": native_resolution,
        "native_resolution_m": native_resolution_m,
        "native_resolution_unit": _infer_resolution_unit(
            source_cfg,
            native_resolution_m=native_resolution_m,
        ),
        "target_resolution_m": processing.get("target_resolution_m"),
    }


def build_feature_metadata(
    source_cfg: dict,
    variable: str,
    variable_cfg: dict,
    aggregation_cfg: dict,
    months: list[int],
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
    resampling_method_name: str,
) -> dict[str, Any]:
    """
    Build metadata for temporal aggregated feature rasters.
    """
    source_meta = _source_metadata(source_cfg)
    dataset_cfg = source_cfg.get("dataset", {})
    native_resolution_m = variable_cfg.get(
        "native_resolution_m",
        dataset_cfg.get("native_resolution_m"),
    )

    metric = (
        aggregation_cfg.get("metric")
        or aggregation_cfg.get("output_metric_name")
        or aggregation_cfg.get("aggregation_metric")
    )

    metadata = {
        **source_meta,
        "variable": variable,
        "variable_description": variable_cfg.get("description"),
        "unit": variable_cfg.get("unit"),
        "valid_range": variable_cfg.get("valid_range"),
        "data_type": variable_cfg.get("data_type"),
        "value_semantics": variable_cfg.get("value_semantics"),
        "scale_factor": variable_cfg.get("scale_factor", 1.0),
        "native_resolution_m": native_resolution_m,
        "native_resolution_unit": _infer_resolution_unit(
            source_cfg,
            native_resolution_m=native_resolution_m,
        ),
        "aggregation_name": aggregation_cfg.get("name"),
        "aggregation_metric": metric,
        "metric": metric,
        "months": months,
        "month_start": min(months) if months else None,
        "month_end": max(months) if months else None,
        "clip_aoi_name": clip_aoi_name,
        "output_aoi_name": output_aoi_name,
        "target_resolution_m": target_resolution_m,
        "resampling": resampling_method_name,
        "resampling_effective_method": (
            "average+area_ratio"
            if is_conservative_resampling(resampling_method_name)
            else resampling_method_name
        ),
    }

    return _clean_metadata(metadata)


def build_static_feature_metadata(
    source_cfg: dict,
    layer_name: str,
    layer_cfg: dict,
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
    resampling_method_name: str,
) -> dict[str, Any]:
    """
    Build metadata for static feature rasters.
    """
    source_meta = _source_metadata(source_cfg)
    dataset_cfg = source_cfg.get("dataset", {})
    native_resolution_m = layer_cfg.get(
        "native_resolution_m",
        dataset_cfg.get("native_resolution_m"),
    )

    metadata = {
        **source_meta,
        "variable": layer_name,
        "variable_description": layer_cfg.get("description"),
        "unit": layer_cfg.get("unit"),
        "valid_range": layer_cfg.get("valid_range"),
        "data_type": layer_cfg.get("data_type"),
        "value_semantics": layer_cfg.get("value_semantics"),
        "scale_factor": layer_cfg.get("scale_factor", 1.0),
        "native_resolution_m": native_resolution_m,
        "native_resolution_unit": _infer_resolution_unit(
            source_cfg,
            native_resolution_m=native_resolution_m,
        ),
        "clip_aoi_name": clip_aoi_name,
        "output_aoi_name": output_aoi_name,
        "target_resolution_m": target_resolution_m,
        "resampling": resampling_method_name,
        "resampling_effective_method": (
            "average+area_ratio"
            if is_conservative_resampling(resampling_method_name)
            else resampling_method_name
        ),
    }

    return _clean_metadata(metadata)


# =============================================================================
# Raster writing
# =============================================================================


def build_output_profile(
    grid: GridContext,
    output_dtype: str = "float32",
    nodata: float = -9999.0,
    compression: str = "LZW",
) -> dict[str, Any]:
    """
    Build a GeoTIFF output profile matching the target grid.
    """
    profile = grid.profile.copy()

    profile.update(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype=output_dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress=compression,
    )

    return profile


def prepare_array_for_write(
    array: np.ndarray,
    nodata: float = -9999.0,
    output_dtype: str = "float32",
) -> np.ndarray:
    """
    Convert an internal float array with np.nan into a writable raster array.
    """
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array for raster writing, got shape {array.shape}"
        )

    out = array.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = nodata

    return out.astype(output_dtype)


def write_feature_raster(
    output_path: Path,
    array: np.ndarray,
    grid: GridContext,
    metadata: dict[str, Any],
    output_dtype: str = "float32",
    nodata: float = -9999.0,
    compression: str = "LZW",
    write_sidecar: bool = True,
    validate: bool = True,
) -> Path:
    """
    Write one feature raster aligned to the project grid.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = build_output_profile(
        grid=grid,
        output_dtype=output_dtype,
        nodata=nodata,
        compression=compression,
    )
    metadata = enrich_output_metadata(
        metadata=metadata,
        output_path=output_path,
        grid=grid,
        output_dtype=output_dtype,
        nodata=nodata,
        compression=compression,
    )

    writable = prepare_array_for_write(
        array=array,
        nodata=nodata,
        output_dtype=output_dtype,
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(writable, 1)
        dst.update_tags(**metadata_to_geotiff_tags(metadata))

    if write_sidecar:
        write_sidecar_json(metadata, output_path)

    if validate:
        validate_raster_matches_grid(
            raster_path=output_path,
            grid_path=grid.path,
        )

    return output_path


# =============================================================================
# Validation
# =============================================================================


def validate_raster_matches_grid(
    raster_path: str | Path,
    grid_path: str | Path,
    check_transform: bool = True,
    check_crs: bool = True,
    check_shape: bool = True,
) -> None:
    """
    Validate that a raster matches the target grid geometry.

    Checks:
      - CRS
      - width/height
      - affine transform
    """
    raster_path = Path(raster_path)
    grid_path = Path(grid_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster does not exist: {raster_path}")

    if not grid_path.exists():
        raise FileNotFoundError(f"Grid does not exist: {grid_path}")

    with rasterio.open(raster_path) as raster, rasterio.open(grid_path) as grid:
        if check_crs and raster.crs != grid.crs:
            raise ValueError(
                f"CRS mismatch for {raster_path}: "
                f"raster={raster.crs}, grid={grid.crs}"
            )

        if check_shape and (
            raster.width != grid.width or raster.height != grid.height
        ):
            raise ValueError(
                f"Shape mismatch for {raster_path}: "
                f"raster=({raster.width}, {raster.height}), "
                f"grid=({grid.width}, {grid.height})"
            )

        if check_transform and raster.transform != grid.transform:
            raise ValueError(
                f"Transform mismatch for {raster_path}: "
                f"raster={raster.transform}, grid={grid.transform}"
            )
