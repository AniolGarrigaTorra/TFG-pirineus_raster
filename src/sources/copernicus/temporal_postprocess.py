from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math
import re
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.warp import transform_bounds

from src.sources.copernicus.postprocess import (
    find_tif_members_in_zip,
    zip_member_to_rasterio_uri,
)


@dataclass(frozen=True)
class TemporalRaster:
    uri: str
    date: datetime
    source_path: str


@dataclass(frozen=True)
class ReferenceGrid:
    crs: Any
    transform: Any
    width: int
    height: int
    profile: dict


def _parse_date_from_text(
    text: str,
    date_regex: str,
    date_format: str,
) -> datetime | None:
    match = re.compile(date_regex).search(text)
    if not match:
        return None

    date_text = match.group(1) if match.groups() else match.group(0)
    return datetime.strptime(date_text, date_format)


def _collect_temporal_rasters_from_zip(
    *,
    zip_path: Path,
    zip_member_pattern: str | None,
    date_regex: str,
    date_format: str,
) -> list[TemporalRaster]:
    members = find_tif_members_in_zip(
        zip_path=zip_path,
        zip_member_pattern=zip_member_pattern,
    )

    rasters: list[TemporalRaster] = []

    for member in members:
        date = _parse_date_from_text(
            text=member,
            date_regex=date_regex,
            date_format=date_format,
        )

        if date is None:
            date = _parse_date_from_text(
                text=zip_path.name,
                date_regex=date_regex,
                date_format=date_format,
            )

        if date is None:
            print(f"[temporal] Could not parse date from {zip_path}!{member}. Skipping.")
            continue

        rasters.append(
            TemporalRaster(
                uri=zip_member_to_rasterio_uri(zip_path, member),
                date=date,
                source_path=f"{zip_path}!{member}",
            )
        )

    return rasters


def _collect_temporal_rasters_from_geotiff(
    *,
    path: Path,
    date_regex: str,
    date_format: str,
) -> list[TemporalRaster]:
    date = _parse_date_from_text(
        text=path.name,
        date_regex=date_regex,
        date_format=date_format,
    )

    if date is None:
        print(f"[temporal] Could not parse date from {path}. Skipping.")
        return []

    return [
        TemporalRaster(
            uri=str(path),
            date=date,
            source_path=str(path),
        )
    ]


def collect_temporal_rasters(
    *,
    input_paths: list[Path],
    temporal_cfg: dict[str, Any],
) -> list[TemporalRaster]:
    zip_member_pattern = temporal_cfg.get("zip_member_pattern")
    date_regex = temporal_cfg["date_regex"]
    date_format = temporal_cfg.get("date_format", "%Y%m%d")

    rasters: list[TemporalRaster] = []

    for path in input_paths:
        path = Path(path)

        if path.suffix.lower() == ".zip":
            rasters.extend(
                _collect_temporal_rasters_from_zip(
                    zip_path=path,
                    zip_member_pattern=zip_member_pattern,
                    date_regex=date_regex,
                    date_format=date_format,
                )
            )
        elif path.suffix.lower() in {".tif", ".tiff"}:
            rasters.extend(
                _collect_temporal_rasters_from_geotiff(
                    path=path,
                    date_regex=date_regex,
                    date_format=date_format,
                )
            )
        else:
            print(f"[temporal] Unsupported temporal input file: {path}. Skipping.")

    return sorted(rasters, key=lambda item: (item.date, item.source_path))


def _filter_by_months(
    rasters: list[TemporalRaster],
    months: list[int] | None,
) -> list[TemporalRaster]:
    if not months:
        return rasters

    months_set = {int(month) for month in months}
    return [raster for raster in rasters if raster.date.month in months_set]


def _filter_by_date_range(
    rasters: list[TemporalRaster],
    start_date: str | None,
    end_date: str | None,
) -> list[TemporalRaster]:
    if start_date is None and end_date is None:
        return rasters

    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None

    selected: list[TemporalRaster] = []

    for raster in rasters:
        if start is not None and raster.date < start:
            continue
        if end is not None and raster.date > end:
            continue
        selected.append(raster)

    return selected


def _group_by_date(rasters: list[TemporalRaster]) -> dict[datetime, list[TemporalRaster]]:
    by_date: dict[datetime, list[TemporalRaster]] = defaultdict(list)

    for raster in rasters:
        by_date[raster.date].append(raster)

    return by_date


def _get_vrt_resampling(name: str | None):
    name = (name or "nearest").lower()

    mapping = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }

    if name not in mapping:
        raise ValueError(f"Unsupported temporal VRT resampling={name!r}")

    return mapping[name]


def _clean_array(
    array: np.ndarray,
    *,
    nodata: float | int | None,
    value_filter: dict[str, Any],
) -> np.ndarray:
    data = array.astype(np.float32, copy=False)

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    nodata_values = value_filter.get("nodata_values", []) or []
    for value in nodata_values:
        data = np.where(data == float(value), np.nan, data)

    valid_range = value_filter.get("valid_range")
    if valid_range is not None:
        low, high = float(valid_range[0]), float(valid_range[1])
        data = np.where((data < low) | (data > high), np.nan, data)

    scale_factor = float(value_filter.get("scale_factor", 1.0))
    if scale_factor != 1.0:
        data = data * scale_factor

    return data.astype(np.float32, copy=False)


def _align_bounds_to_resolution(
    *,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    resolution: float,
) -> tuple[float, float, float, float]:
    return (
        math.floor(minx / resolution) * resolution,
        math.floor(miny / resolution) * resolution,
        math.ceil(maxx / resolution) * resolution,
        math.ceil(maxy / resolution) * resolution,
    )


def _build_reference_grid_from_config(
    *,
    temporal_cfg: dict[str, Any],
    first_profile: dict,
) -> ReferenceGrid | None:
    reference_cfg = temporal_cfg.get("reference_grid") or {}
    bounds_cfg = reference_cfg.get("bounds")

    if not bounds_cfg:
        return None

    crs = reference_cfg.get("crs") or temporal_cfg.get("target_crs")
    resolution = float(
        reference_cfg.get("resolution_m")
        or temporal_cfg.get("target_resolution_m")
    )

    minx = float(bounds_cfg["xmin"])
    maxx = float(bounds_cfg["xmax"])
    miny = float(bounds_cfg["ymin"])
    maxy = float(bounds_cfg["ymax"])

    minx, miny, maxx, maxy = _align_bounds_to_resolution(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        resolution=resolution,
    )

    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    profile = first_profile.copy()
    profile.update(
        driver="GTiff",
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        count=1,
        dtype="float32",
        BIGTIFF="IF_SAFER",
    )

    print("[temporal] Reference grid from config:")
    print(f"  CRS: {crs}")
    print(f"  Resolution: {resolution}")
    print(f"  Width x height: {width} x {height}")
    print(f"  Bounds: {minx}, {miny}, {maxx}, {maxy}")

    return ReferenceGrid(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        profile=profile,
    )


def _build_reference_grid_from_rasters(
    *,
    rasters: list[TemporalRaster],
    temporal_cfg: dict[str, Any],
    first_profile: dict,
) -> ReferenceGrid:
    target_crs = temporal_cfg.get("target_crs")
    target_resolution_m = temporal_cfg.get("target_resolution_m")

    if target_resolution_m is None:
        raise ValueError(
            "temporal_postprocess.target_resolution_m is required for temporal aggregation."
        )

    resolution = float(target_resolution_m)

    minx_values: list[float] = []
    miny_values: list[float] = []
    maxx_values: list[float] = []
    maxy_values: list[float] = []

    for raster in rasters:
        with rasterio.open(raster.uri) as src:
            if target_crs is None:
                target_crs = src.crs

            bounds = transform_bounds(
                src.crs,
                target_crs,
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
                densify_pts=21,
            )

            minx_values.append(bounds[0])
            miny_values.append(bounds[1])
            maxx_values.append(bounds[2])
            maxy_values.append(bounds[3])

    minx, miny, maxx, maxy = _align_bounds_to_resolution(
        minx=min(minx_values),
        miny=min(miny_values),
        maxx=max(maxx_values),
        maxy=max(maxy_values),
        resolution=resolution,
    )

    width = int(math.ceil((maxx - minx) / resolution))
    height = int(math.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    profile = first_profile.copy()
    profile.update(
        driver="GTiff",
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        count=1,
        dtype="float32",
        BIGTIFF="IF_SAFER",
    )

    print("[temporal] Reference grid from raster union:")
    print(f"  CRS: {target_crs}")
    print(f"  Resolution: {resolution}")
    print(f"  Width x height: {width} x {height}")
    print(f"  Bounds: {minx}, {miny}, {maxx}, {maxy}")

    return ReferenceGrid(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        profile=profile,
    )


def _build_reference_grid(
    *,
    rasters: list[TemporalRaster],
    temporal_cfg: dict[str, Any],
) -> ReferenceGrid:
    if not rasters:
        raise ValueError("Cannot build reference grid from empty rasters list.")

    with rasterio.open(rasters[0].uri) as src:
        first_profile = src.profile.copy()

    configured = _build_reference_grid_from_config(
        temporal_cfg=temporal_cfg,
        first_profile=first_profile,
    )

    if configured is not None:
        return configured

    return _build_reference_grid_from_rasters(
        rasters=rasters,
        temporal_cfg=temporal_cfg,
        first_profile=first_profile,
    )


def _iter_windows(
    *,
    width: int,
    height: int,
    block_size: int,
):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)

        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=win_w,
                height=win_h,
            )


def _read_mosaic_for_date_window(
    rasters: list[TemporalRaster],
    *,
    temporal_cfg: dict[str, Any],
    reference_grid: ReferenceGrid,
    window: Window,
) -> np.ndarray:
    """
    Read one or more rasters for the same date into one window of the common
    reference grid.

    If several Sentinel-2 tiles exist for the same date, overlapping valid
    pixels are averaged.

    Important:
    WarpedVRT does not allow boundless=True reads. This is fine here because
    the window is always inside the WarpedVRT reference grid dimensions.
    """
    vrt_resampling = _get_vrt_resampling(temporal_cfg.get("vrt_resampling", "nearest"))
    value_filter = temporal_cfg.get("value_filter", {}) or {}

    shape = (int(window.height), int(window.width))

    sum_array = np.zeros(shape, dtype=np.float32)
    count_array = np.zeros(shape, dtype=np.float32)

    for raster in rasters:
        with rasterio.open(raster.uri) as src:
            with WarpedVRT(
                src,
                crs=reference_grid.crs,
                transform=reference_grid.transform,
                width=reference_grid.width,
                height=reference_grid.height,
                resampling=vrt_resampling,
            ) as vrt:
                masked_data = vrt.read(
                    1,
                    window=window,
                    masked=True,
                )

                if np.ma.isMaskedArray(masked_data):
                    data = masked_data.astype(np.float32).filled(np.nan)
                else:
                    data = masked_data.astype(np.float32)

                cleaned = _clean_array(
                    data,
                    nodata=vrt.nodata,
                    value_filter=value_filter,
                )

        valid = np.isfinite(cleaned)
        sum_array[valid] += cleaned[valid]
        count_array[valid] += 1.0

    output = np.full(shape, np.nan, dtype=np.float32)

    valid_count = count_array > 0
    output[valid_count] = sum_array[valid_count] / count_array[valid_count]

    return output


def _comparison_mask(
    array: np.ndarray,
    threshold: float,
    comparison: str,
) -> np.ndarray:
    if comparison == ">=":
        return array >= threshold
    if comparison == ">":
        return array > threshold
    if comparison == "<=":
        return array <= threshold
    if comparison == "<":
        return array < threshold
    if comparison == "==":
        return array == threshold

    raise ValueError(f"Unsupported comparison: {comparison!r}")


def _initial_metric_state(
    *,
    method: str,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    if method in {"mean", "std"}:
        return {
            "sum": np.zeros(shape, dtype=np.float32),
            "count": np.zeros(shape, dtype=np.float32),
            "sumsq": np.zeros(shape, dtype=np.float32),
        }

    if method == "valid_observation_count":
        return {
            "count": np.zeros(shape, dtype=np.float32),
        }

    if method == "count_threshold":
        return {
            "count": np.zeros(shape, dtype=np.float32),
        }

    if method == "min":
        return {
            "value": np.full(shape, np.nan, dtype=np.float32),
        }

    if method == "max":
        return {
            "value": np.full(shape, np.nan, dtype=np.float32),
        }

    raise NotImplementedError(
        f"Unsupported streaming temporal metric method={method!r}. "
        "Supported methods: mean, std, min, max, count_threshold, valid_observation_count."
    )


def _update_metric_state(
    *,
    state: dict[str, np.ndarray],
    array: np.ndarray,
    metric_cfg: dict[str, Any],
) -> None:
    method = metric_cfg["method"]
    valid = np.isfinite(array)

    if method == "mean":
        state["sum"][valid] += array[valid]
        state["count"][valid] += 1.0
        return

    if method == "std":
        state["sum"][valid] += array[valid]
        state["sumsq"][valid] += array[valid] ** 2
        state["count"][valid] += 1.0
        return

    if method == "valid_observation_count":
        state["count"] += valid.astype(np.float32)
        return

    if method == "count_threshold":
        threshold = float(metric_cfg["threshold"])
        comparison = str(metric_cfg.get("comparison", ">="))
        mask = valid & _comparison_mask(array, threshold, comparison)
        state["count"] += mask.astype(np.float32)
        return

    if method == "min":
        current = state["value"]
        replace = valid & (~np.isfinite(current) | (array < current))
        current[replace] = array[replace]
        return

    if method == "max":
        current = state["value"]
        replace = valid & (~np.isfinite(current) | (array > current))
        current[replace] = array[replace]
        return

    raise NotImplementedError(f"Unsupported metric method={method!r}")


def _finalize_metric_state(
    *,
    state: dict[str, np.ndarray],
    metric_cfg: dict[str, Any],
) -> np.ndarray:
    method = metric_cfg["method"]

    if method == "mean":
        result = np.full_like(state["sum"], np.nan, dtype=np.float32)
        valid = state["count"] > 0
        result[valid] = state["sum"][valid] / state["count"][valid]
        return result

    if method == "std":
        result = np.full_like(state["sum"], np.nan, dtype=np.float32)
        valid = state["count"] > 0
        mean = np.zeros_like(state["sum"], dtype=np.float32)
        mean[valid] = state["sum"][valid] / state["count"][valid]
        variance = np.zeros_like(state["sum"], dtype=np.float32)
        variance[valid] = state["sumsq"][valid] / state["count"][valid] - mean[valid] ** 2
        variance = np.where(variance < 0, 0, variance)
        result[valid] = np.sqrt(variance[valid])
        return result.astype(np.float32)

    if method in {"valid_observation_count", "count_threshold"}:
        return state["count"].astype(np.float32)

    if method in {"min", "max"}:
        return state["value"].astype(np.float32)

    raise NotImplementedError(f"Unsupported metric method={method!r}")


def _build_write_profile(
    *,
    reference_grid: ReferenceGrid,
    compression: str,
) -> dict[str, Any]:
    profile = reference_grid.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress=compression,
        BIGTIFF="IF_SAFER",
    )
    return profile


def _write_tags(
    dst,
    metadata: dict[str, Any],
) -> None:
    dst.update_tags(
        **{
            key: str(value)
            for key, value in metadata.items()
            if value is not None
        }
    )


def _compute_and_write_metric_windowed(
    *,
    output_path: Path,
    selected_by_date: dict[datetime, list[TemporalRaster]],
    temporal_cfg: dict[str, Any],
    metric_cfg: dict[str, Any],
    reference_grid: ReferenceGrid,
    compression: str,
    overwrite: bool,
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[temporal] Exists, skipping: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and overwrite:
        output_path.unlink()

    method = metric_cfg["method"]
    block_size = int(temporal_cfg.get("block_size", 512))

    profile = _build_write_profile(
        reference_grid=reference_grid,
        compression=compression,
    )

    windows = list(
        _iter_windows(
            width=reference_grid.width,
            height=reference_grid.height,
            block_size=block_size,
        )
    )

    print(f"[temporal] Writing windowed output: {output_path}")
    print(f"[temporal] Method: {method}")
    print(f"[temporal] Block size: {block_size}")
    print(f"[temporal] Windows: {len(windows)}")

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, window in enumerate(windows, start=1):
            shape = (int(window.height), int(window.width))

            state = _initial_metric_state(
                method=method,
                shape=shape,
            )

            for date in sorted(selected_by_date):
                date_array = _read_mosaic_for_date_window(
                    selected_by_date[date],
                    temporal_cfg=temporal_cfg,
                    reference_grid=reference_grid,
                    window=window,
                )

                _update_metric_state(
                    state=state,
                    array=date_array,
                    metric_cfg=metric_cfg,
                )

            result = _finalize_metric_state(
                state=state,
                metric_cfg=metric_cfg,
            )

            dst.write(result.astype(np.float32), 1, window=window)

            if idx == 1 or idx % 50 == 0 or idx == len(windows):
                print(f"[temporal] Window {idx}/{len(windows)} written")

        _write_tags(dst, metadata)

    print(f"[temporal] Written: {output_path}")
    return output_path


def _run_export_timesteps(
    *,
    by_date: dict[datetime, list[TemporalRaster]],
    raw_dir: Path,
    source_cfg: dict[str, Any],
    temporal_cfg: dict[str, Any],
) -> list[Path]:
    export_cfg = temporal_cfg.get("export_timesteps", {}) or {}

    if not bool(export_cfg.get("enabled", False)):
        return []

    raise NotImplementedError(
        "export_timesteps is intentionally disabled for now in the windowed "
        "temporal engine. Enable later with a dedicated windowed export path."
    )


def run_temporal_zip_geotiff_aggregation(
    *,
    input_paths: list[Path],
    raw_dir: Path,
    source_cfg: dict[str, Any],
    spec: dict[str, Any],
) -> list[Path]:
    temporal_cfg = source_cfg.get("temporal_postprocess", {}) or {}
    output_variables = temporal_cfg.get("output_variables", {}) or {}

    if not output_variables:
        raise ValueError(
            "temporal_zip_geotiff_aggregation requires "
            "temporal_postprocess.output_variables or output_variable_groups."
        )

    download_cfg = source_cfg.get("download", {}) or {}
    output_cfg = source_cfg.get("output", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    compression = str(output_cfg.get("compression", "LZW"))

    rasters = collect_temporal_rasters(
        input_paths=input_paths,
        temporal_cfg=temporal_cfg,
    )

    if not rasters:
        raise FileNotFoundError(
            "No temporal rasters were collected. Check date_regex, "
            "zip_member_pattern and downloaded files."
        )

    rasters = _filter_by_date_range(
        rasters,
        start_date=temporal_cfg.get("start_date"),
        end_date=temporal_cfg.get("end_date"),
    )

    if not rasters:
        raise FileNotFoundError(
            "No temporal rasters remained after start_date/end_date filtering."
        )

    by_date = _group_by_date(rasters)

    print("[temporal] Temporal rasters collected:", len(rasters))
    print("[temporal] Unique dates:", len(by_date))

    outputs: list[Path] = []

    generated_manifest: dict[str, Any] = {
        "source_id": source_cfg.get("source", {}).get("id"),
        "product": source_cfg.get("source", {}).get("product"),
        "postprocess": "temporal_zip_geotiff_aggregation_windowed",
        "n_input_files": len(input_paths),
        "n_temporal_rasters": len(rasters),
        "n_dates": len(by_date),
        "generated_variables": {},
    }

    for variable_name, metric_cfg in output_variables.items():
        months = metric_cfg.get("months")

        selected_rasters = _filter_by_months(rasters, months)
        selected_rasters = _filter_by_date_range(
            selected_rasters,
            start_date=metric_cfg.get("start_date"),
            end_date=metric_cfg.get("end_date"),
        )

        if not selected_rasters:
            print(
                f"[temporal] No rasters selected for variable={variable_name!r}, "
                f"months={months}. Skipping."
            )
            continue

        selected_by_date = _group_by_date(selected_rasters)

        print("==============================")
        print(f"[temporal] Variable: {variable_name}")
        print(f"[temporal] Method: {metric_cfg.get('method')}")
        print(f"[temporal] Months: {months}")
        print(f"[temporal] Dates selected: {len(selected_by_date)}")

        reference_grid = _build_reference_grid(
            rasters=selected_rasters,
            temporal_cfg=temporal_cfg,
        )

        filename = metric_cfg.get("filename", f"{variable_name}.tif")
        output_path = Path(raw_dir) / filename

        written = _compute_and_write_metric_windowed(
            output_path=output_path,
            selected_by_date=selected_by_date,
            temporal_cfg=temporal_cfg,
            metric_cfg=metric_cfg,
            reference_grid=reference_grid,
            compression=compression,
            overwrite=overwrite,
            metadata={
                "variable": variable_name,
                "method": metric_cfg.get("method"),
                "months": months,
                "threshold": metric_cfg.get("threshold"),
                "comparison": metric_cfg.get("comparison"),
                "n_dates": len(selected_by_date),
                "postprocess": "temporal_zip_geotiff_aggregation_windowed",
                "target_crs": reference_grid.crs,
                "target_resolution_m": temporal_cfg.get("target_resolution_m"),
            },
        )

        generated_manifest["generated_variables"][variable_name] = {
            "path": str(written),
            "filename": filename,
            "method": metric_cfg.get("method"),
            "months": months,
            "n_dates": len(selected_by_date),
            "unit": metric_cfg.get("unit"),
            "valid_range": metric_cfg.get("valid_range"),
            "native_resolution_m": metric_cfg.get("native_resolution_m"),
        }

        outputs.append(written)

    timestep_outputs = _run_export_timesteps(
        by_date=by_date,
        raw_dir=Path(raw_dir),
        source_cfg=source_cfg,
        temporal_cfg=temporal_cfg,
    )

    if timestep_outputs:
        generated_manifest["timestep_outputs"] = [str(path) for path in timestep_outputs]
        outputs.extend(timestep_outputs)

    manifest_path = Path(raw_dir) / "temporal_generated_variables.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(generated_manifest, f, indent=2, ensure_ascii=False)

    print(f"[temporal] Manifest written: {manifest_path}")

    if not outputs:
        raise RuntimeError(
            "Temporal postprocess finished but produced no outputs. "
            "Check temporal months/date filters."
        )

    return outputs