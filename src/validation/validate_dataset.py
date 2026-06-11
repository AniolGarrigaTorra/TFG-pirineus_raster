from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rasterio

from src.io.config import load_yaml, resolve_path
from src.io.paths import get_grid_path


COMMON_METADATA_KEYS = {
    "metadata_schema_version",
    "generated_at",
    "provider",
    "product",
    "source_id",
    "variable",
    "output_crs",
    "output_resolution_m",
    "nodata",
    "dtype",
    "grid_path",
}

SOURCE_METADATA_KEYS = {
    "native_resolution",
    "native_resolution_unit",
    "source_config_path",
    "source_config_sha256",
    "target_resolution_m",
    "resampling",
}

DERIVED_METADATA_KEYS = {
    "operation",
    "inputs",
}


def _required_metadata_keys(metadata: dict[str, Any]) -> set[str]:
    required = set(COMMON_METADATA_KEYS)
    if metadata.get("source_id") == "derived" or metadata.get("layer_type") == "derived":
        required.update(DERIVED_METADATA_KEYS)
    else:
        required.update(SOURCE_METADATA_KEYS)
    return required


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _resolve_manifest_path(value: str | None, manifest_path: Path) -> Path | None:
    if not value:
        return None
    return resolve_path(value, base_path=manifest_path, must_exist=True)


def _load_project_cfg(manifest: dict[str, Any], manifest_path: Path) -> tuple[Path, dict]:
    project_config_path = _resolve_manifest_path(
        manifest.get("project_config"),
        manifest_path,
    )

    if project_config_path is None:
        project_config_path = resolve_path("configs/project.yaml", must_exist=True)

    project_cfg = load_yaml(project_config_path)
    project_cfg["_config_path"] = str(project_config_path)

    return project_config_path, project_cfg


def _load_aoi_cfg(manifest: dict[str, Any], manifest_path: Path) -> tuple[Path, dict]:
    run_aoi_config = manifest.get("run_aoi_config")
    run_aoi_name = manifest.get("run_aoi_name")

    if run_aoi_config:
        aoi_config_path = _resolve_manifest_path(run_aoi_config, manifest_path)
    elif run_aoi_name:
        aoi_config_path = resolve_path(
            Path("configs/aoi") / f"{run_aoi_name}.yaml",
            base_path=manifest_path,
            must_exist=True,
        )
    else:
        raise ValueError(
            "Manifest has neither run_aoi_config nor run_aoi_name. "
            "Cannot locate reference grid."
        )

    if aoi_config_path is None:
        raise FileNotFoundError("Could not resolve AOI config from manifest.")

    return aoi_config_path, load_yaml(aoi_config_path)


def _raster_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in manifest.get("rasters", [])
        if str(entry.get("dataset_path", "")).lower().endswith((".tif", ".tiff"))
    ]


def _same_transform(left, right) -> bool:
    return tuple(left) == tuple(right)


def _metadata_warnings(
    sidecar_path: Path | None,
    strict_metadata: bool,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if sidecar_path is None or not sidecar_path.exists():
        message = "Missing sidecar metadata JSON."
        if strict_metadata:
            errors.append(message)
        else:
            warnings.append(message)
        return None, warnings, errors

    try:
        metadata = _read_json(sidecar_path)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid sidecar JSON: {exc}")
        return None, warnings, errors

    missing = sorted(_required_metadata_keys(metadata) - set(metadata))
    if missing:
        message = f"Missing standard metadata keys: {missing}"
        if strict_metadata:
            errors.append(message)
        else:
            warnings.append(message)

    return metadata, warnings, errors


def validate_one_raster(
    raster_entry: dict[str, Any],
    dataset_dir: Path,
    grid,
    strict_metadata: bool = False,
) -> dict[str, Any]:
    raster_path = resolve_path(
        raster_entry["dataset_path"],
        base_path=dataset_dir,
        must_exist=True,
    )

    sidecar_value = raster_entry.get("sidecar_json_dataset_path")
    if sidecar_value:
        try:
            sidecar_path = resolve_path(
                sidecar_value,
                base_path=dataset_dir,
                must_exist=True,
            )
        except FileNotFoundError:
            sidecar_path = Path(sidecar_value)
    else:
        sidecar_path = raster_path.with_suffix(".json")

    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []

    with rasterio.open(raster_path) as src:
        checks["crs_match"] = src.crs == grid.crs
        checks["shape_match"] = src.width == grid.width and src.height == grid.height
        checks["transform_match"] = _same_transform(src.transform, grid.transform)
        checks["single_band"] = src.count == 1

        if src.nodata != grid.nodata:
            warnings.append(
                f"NoData differs from grid: raster={src.nodata}, grid={grid.nodata}"
            )

        summary = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0] if src.dtypes else None,
            "nodata": src.nodata,
            "transform": list(src.transform),
            "bounds": {
                "xmin": float(src.bounds.left),
                "ymin": float(src.bounds.bottom),
                "xmax": float(src.bounds.right),
                "ymax": float(src.bounds.top),
            },
        }

    for key, ok in checks.items():
        if not ok:
            errors.append(f"Failed check: {key}")

    _, metadata_warnings, metadata_errors = _metadata_warnings(
        sidecar_path=sidecar_path,
        strict_metadata=strict_metadata,
    )
    warnings.extend(metadata_warnings)
    errors.extend(metadata_errors)

    return {
        "name": raster_entry.get("name") or raster_path.stem,
        "path": str(raster_path),
        "sidecar_path": str(sidecar_path) if sidecar_path else None,
        "checks": checks,
        "summary": summary,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }


def validate_dataset_dir(
    dataset_dir: str | Path,
    strict_metadata: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    dataset_dir = resolve_path(dataset_dir, must_exist=True)
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    project_config_path, project_cfg = _load_project_cfg(manifest, manifest_path)
    aoi_config_path, aoi_cfg = _load_aoi_cfg(manifest, manifest_path)

    resolution_m = manifest.get("run_resolution_m")
    if resolution_m is None:
        raise ValueError("Manifest is missing run_resolution_m.")
    resolution_m = int(resolution_m)

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(f"Reference grid not found: {grid_path}")

    raster_entries = _raster_entries(manifest)
    raster_results: list[dict[str, Any]] = []

    with rasterio.open(grid_path) as grid:
        for raster_entry in raster_entries:
            try:
                result = validate_one_raster(
                    raster_entry=raster_entry,
                    dataset_dir=dataset_dir,
                    grid=grid,
                    strict_metadata=strict_metadata,
                )
            except Exception as exc:
                result = {
                    "name": raster_entry.get("name"),
                    "path": raster_entry.get("dataset_path"),
                    "checks": {},
                    "summary": {},
                    "warnings": [],
                    "errors": [str(exc)],
                    "ok": False,
                }

            raster_results.append(result)

    failed = [item for item in raster_results if not item["ok"]]
    warned = [item for item in raster_results if item.get("warnings")]

    report = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "project_config_path": str(project_config_path),
        "aoi_config_path": str(aoi_config_path),
        "grid_path": str(grid_path),
        "strict_metadata": strict_metadata,
        "n_rasters": len(raster_results),
        "n_failed": len(failed),
        "n_warned": len(warned),
        "ok": len(failed) == 0,
        "rasters": raster_results,
    }

    if write_report:
        _write_json(dataset_dir / "metadata" / "validation_report.json", report)

    return report
