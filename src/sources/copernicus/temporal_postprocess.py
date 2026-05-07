from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

from src.sources.copernicus.postprocess import (
    find_tif_members_in_zip,
    zip_member_to_rasterio_uri,
)


@dataclass(frozen=True)
class TemporalRaster:
    uri: str
    date: datetime
    source_path: str


def _parse_date_from_text(
    text: str,
    date_regex: str,
    date_format: str,
) -> datetime | None:
    pattern = re.compile(date_regex)
    match = pattern.search(text)

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
            print(
                f"[temporal] Could not parse date from {zip_path}!{member}. "
                "Skipping."
            )
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

    return [
        raster
        for raster in rasters
        if raster.date.month in months_set
    ]


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
    data = array.astype(np.float32)

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

    return data


def _read_mosaic_for_date(
    rasters: list[TemporalRaster],
    *,
    temporal_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict]:
    """
    Read one or more rasters for the same date.

    If temporal_postprocess.target_crs is set, all rasters are read through
    WarpedVRT before mosaic. This is important for Sentinel-2 tile grids that
    may cross UTM zones.
    """
    target_crs = temporal_cfg.get("target_crs")
    target_resolution_m = temporal_cfg.get("target_resolution_m")
    vrt_resampling = _get_vrt_resampling(temporal_cfg.get("vrt_resampling", "nearest"))
    value_filter = temporal_cfg.get("value_filter", {}) or {}

    opened = []
    srcs = []

    try:
        for raster in rasters:
            src = rasterio.open(raster.uri)
            opened.append(src)

            if target_crs:
                vrt_kwargs: dict[str, Any] = {
                    "crs": target_crs,
                    "resampling": vrt_resampling,
                }

                if target_resolution_m is not None:
                    vrt_kwargs["x_res"] = float(target_resolution_m)
                    vrt_kwargs["y_res"] = float(target_resolution_m)

                vrt = WarpedVRT(src, **vrt_kwargs)
                srcs.append(vrt)
            else:
                srcs.append(src)

        if len(srcs) == 1:
            src = srcs[0]
            data = src.read(1)
            data = _clean_array(
                data,
                nodata=src.nodata,
                value_filter=value_filter,
            )

            profile = src.profile.copy()
            profile.update(count=1, dtype="float32")
            return data, profile

        mosaic, transform = merge(srcs)
        data = mosaic[0]

        nodata = srcs[0].nodata
        data = _clean_array(
            data,
            nodata=nodata,
            value_filter=value_filter,
        )

        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            transform=transform,
            dtype="float32",
            BIGTIFF="IF_SAFER",
        )

        return data, profile

    finally:
        for src in srcs:
            try:
                src.close()
            except Exception:
                pass

        for src in opened:
            try:
                src.close()
            except Exception:
                pass


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


def _compute_metric(
    *,
    arrays: list[np.ndarray],
    metric_cfg: dict[str, Any],
) -> np.ndarray:
    if not arrays:
        raise ValueError("No arrays available for temporal metric.")

    method = metric_cfg["method"]

    if method in {"mean", "median", "min", "max", "std"}:
        stack = np.stack(arrays, axis=0).astype(np.float32)

        if method == "mean":
            return np.nanmean(stack, axis=0).astype(np.float32)
        if method == "median":
            return np.nanmedian(stack, axis=0).astype(np.float32)
        if method == "min":
            return np.nanmin(stack, axis=0).astype(np.float32)
        if method == "max":
            return np.nanmax(stack, axis=0).astype(np.float32)
        if method == "std":
            return np.nanstd(stack, axis=0).astype(np.float32)

    if method == "valid_observation_count":
        count = np.zeros_like(arrays[0], dtype=np.float32)
        for array in arrays:
            count += np.isfinite(array).astype(np.float32)
        return count

    if method == "count_threshold":
        threshold = float(metric_cfg["threshold"])
        comparison = str(metric_cfg.get("comparison", ">="))

        count = np.zeros_like(arrays[0], dtype=np.float32)

        for array in arrays:
            valid = np.isfinite(array)
            mask = _comparison_mask(array, threshold, comparison)
            count += (valid & mask).astype(np.float32)

        return count

    raise NotImplementedError(f"Unsupported temporal metric method={method!r}")


def _write_temporal_output(
    *,
    output_path: Path,
    array: np.ndarray,
    profile: dict,
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

    write_profile = profile.copy()
    write_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        compress=compression,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(array.astype(np.float32), 1)
        dst.update_tags(
            **{
                key: str(value)
                for key, value in metadata.items()
                if value is not None
            }
        )

    print(f"[temporal] Written: {output_path}")
    return output_path


def _group_by_date(rasters: list[TemporalRaster]) -> dict[datetime, list[TemporalRaster]]:
    by_date: dict[datetime, list[TemporalRaster]] = defaultdict(list)
    for raster in rasters:
        by_date[raster.date].append(raster)
    return by_date


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

    download_cfg = source_cfg.get("download", {}) or {}
    output_cfg = source_cfg.get("output", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    compression = str(output_cfg.get("compression", "LZW"))

    output_dir = Path(raw_dir) / export_cfg.get("output_dir", "timesteps")
    naming = export_cfg.get("naming", "temporal_{date}.tif")

    start_date = export_cfg.get("start_date")
    end_date = export_cfg.get("end_date")

    written_paths: list[Path] = []
    manifest: dict[str, Any] = {
        "type": "temporal_timesteps",
        "generated_variables": {},
    }

    for date in sorted(by_date):
        date_iso = date.strftime("%Y-%m-%d")

        if start_date and date < datetime.fromisoformat(start_date):
            continue
        if end_date and date > datetime.fromisoformat(end_date):
            continue

        array, profile = _read_mosaic_for_date(
            by_date[date],
            temporal_cfg=temporal_cfg,
        )

        variable_name = export_cfg.get("variable_name_pattern", "temporal_{date}")
        variable_name = variable_name.format(
            date=date.strftime("%Y_%m_%d"),
            date_iso=date_iso,
        )

        filename = naming.format(
            variable=variable_name,
            date=date.strftime("%Y_%m_%d"),
            date_iso=date_iso,
        )

        output_path = output_dir / filename

        written = _write_temporal_output(
            output_path=output_path,
            array=array,
            profile=profile,
            compression=compression,
            overwrite=overwrite,
            metadata={
                "variable": variable_name,
                "date": date_iso,
                "postprocess": "temporal_zip_geotiff_export",
            },
        )

        manifest["generated_variables"][variable_name] = {
            "path": str(written),
            "filename": str(Path(export_cfg.get("output_dir", "timesteps")) / filename),
            "date": date_iso,
            "unit": export_cfg.get("unit"),
            "data_type": export_cfg.get("data_type"),
            "native_resolution_m": export_cfg.get("native_resolution_m"),
            "temporal": {
                "type": "timestep",
                "date": date_iso,
            },
        }

        written_paths.append(written)

    manifest_path = output_dir / "temporal_timesteps_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[temporal] Timesteps manifest written: {manifest_path}")
    return written_paths


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
        "postprocess": "temporal_zip_geotiff_aggregation",
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
            raise ValueError(
                f"No temporal rasters selected for variable={variable_name!r}. "
                f"months={months}"
            )

        selected_by_date = _group_by_date(selected_rasters)

        arrays: list[np.ndarray] = []
        reference_profile: dict | None = None

        print("==============================")
        print(f"[temporal] Variable: {variable_name}")
        print(f"[temporal] Method: {metric_cfg.get('method')}")
        print(f"[temporal] Months: {months}")
        print(f"[temporal] Dates selected: {len(selected_by_date)}")

        for date in sorted(selected_by_date):
            array, profile = _read_mosaic_for_date(
                selected_by_date[date],
                temporal_cfg=temporal_cfg,
            )
            arrays.append(array)

            if reference_profile is None:
                reference_profile = profile

        if reference_profile is None:
            raise RuntimeError(f"No reference profile for {variable_name}")

        result = _compute_metric(
            arrays=arrays,
            metric_cfg=metric_cfg,
        )

        filename = metric_cfg.get("filename", f"{variable_name}.tif")
        output_path = Path(raw_dir) / filename

        written = _write_temporal_output(
            output_path=output_path,
            array=result,
            profile=reference_profile,
            compression=compression,
            overwrite=overwrite,
            metadata={
                "variable": variable_name,
                "method": metric_cfg.get("method"),
                "months": months,
                "threshold": metric_cfg.get("threshold"),
                "comparison": metric_cfg.get("comparison"),
                "n_dates": len(selected_by_date),
                "postprocess": "temporal_zip_geotiff_aggregation",
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
        }

        outputs.append(written)

    timestep_outputs = _run_export_timesteps(
        by_date=by_date,
        raw_dir=Path(raw_dir),
        source_cfg=source_cfg,
        temporal_cfg=temporal_cfg,
    )

    if timestep_outputs:
        generated_manifest["timestep_outputs"] = [
            str(path)
            for path in timestep_outputs
        ]

    manifest_path = Path(raw_dir) / "temporal_generated_variables.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(generated_manifest, f, indent=2, ensure_ascii=False)

    print(f"[temporal] Manifest written: {manifest_path}")

    return outputs + timestep_outputs