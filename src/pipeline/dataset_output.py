from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.io.config import load_yaml
from src.pipeline.layer_catalog import (
    build_layer_catalog_from_manifest,
    summarize_layer_catalog,
)
from src.pipeline.layer_spec import layer_specs_to_dicts


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dataset_dirs(
    dataset_dir: Path,
    project_cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """
    Create and return standard dataset output directories.
    """
    datasets_cfg = (project_cfg or {}).get("datasets", {})

    rasters_subdir = datasets_cfg.get("rasters_subdir", "rasters")
    metadata_subdir = datasets_cfg.get("metadata_subdir", "metadata")
    config_subdir = datasets_cfg.get("config_subdir", "config")
    logs_subdir = datasets_cfg.get("logs_subdir", "logs")

    dirs = {
        "root": dataset_dir,
        "rasters": dataset_dir / rasters_subdir,
        "metadata": dataset_dir / metadata_subdir,
        "config": dataset_dir / config_subdir,
        "logs": dataset_dir / logs_subdir,
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f_src:
        content = f_src.read()

    with dst.open("w", encoding="utf-8") as f_dst:
        f_dst.write(content)


def safe_dataset_filename(
    source_id: str,
    source_path: Path,
) -> str:
    """
    Build a stable filename for a copied dataset raster.

    The current WorldClim filenames are already descriptive, so we keep them.
    """
    return source_path.name


def associated_sidecar_json(raster_path: Path) -> Path:
    """
    Return expected sidecar JSON path for a raster.
    """
    return raster_path.with_suffix(".json")


def copy_raster_to_dataset(
    raster_path: Path,
    dataset_rasters_dir: Path,
    source_id: str,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Copy one raster and its sidecar JSON, if present, into the dataset folder.
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file does not exist: {raster_path}")

    dst_name = safe_dataset_filename(
        source_id=source_id,
        source_path=raster_path,
    )
    dst_raster = dataset_rasters_dir / dst_name

    if dst_raster.exists() and not overwrite:
        raise FileExistsError(
            f"Destination raster already exists and overwrite=False: {dst_raster}"
        )

    shutil.copy2(raster_path, dst_raster)

    src_json = associated_sidecar_json(raster_path)
    dst_json = None

    if src_json.exists():
        dst_json = dst_raster.with_suffix(".json")
        if dst_json.exists() and not overwrite:
            raise FileExistsError(
                f"Destination metadata JSON already exists and overwrite=False: {dst_json}"
            )
        shutil.copy2(src_json, dst_json)

    return {
        "name": dst_raster.stem,
        "source_id": source_id,
        "original_path": str(raster_path),
        "dataset_path": str(dst_raster),
        "sidecar_json_original_path": str(src_json) if src_json.exists() else None,
        "sidecar_json_dataset_path": str(dst_json) if dst_json is not None else None,
    }


def copy_stage_rasters_to_dataset(
    source_id: str,
    stage_results: list[dict[str, Any]],
    dataset_rasters_dir: Path,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """
    Copy rasters from build stage results to the dataset rasters directory.

    Only .tif files from stage='build' are copied.
    """
    copied: list[dict[str, Any]] = []

    for stage_result in stage_results:
        if stage_result.get("stage") != "build":
            continue

        for raw_path in stage_result.get("paths", []):
            path = Path(raw_path)

            if path.suffix.lower() not in {".tif", ".tiff"}:
                continue

            copied.append(
                copy_raster_to_dataset(
                    raster_path=path,
                    dataset_rasters_dir=dataset_rasters_dir,
                    source_id=source_id,
                    overwrite=overwrite,
                )
            )

    return copied


def build_manifest(
    run_name: str,
    run_config_path: Path,
    project_config_path: Path,
    dataset_dir: Path,
    source_results: list[dict[str, Any]],
    copied_rasters: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    run_aoi_config_path: Path | None = None,
    run_aoi_name: str | None = None,
    run_resolution_m: int | None = None,
) -> dict[str, Any]:
    """
    Build the dataset manifest.

    The manifest is the main index of the final dataset.
    """
    base_manifest: dict[str, Any] = {
        "schema_version": "0.2",
        "dataset_name": run_name,
        "dataset_dir": str(dataset_dir),
        "created_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "run_config": str(run_config_path),
        "project_config": str(project_config_path),
        "n_sources": len(source_results),
        "n_rasters": len(copied_rasters),
        "sources": source_results,
        "rasters": copied_rasters,
        "run_aoi_config": str(run_aoi_config_path) if run_aoi_config_path else None,
        "run_aoi_name": run_aoi_name,
        "run_resolution_m": run_resolution_m,
    }

    layers = build_layer_catalog_from_manifest(base_manifest)
    layer_summary = summarize_layer_catalog(layers)

    base_manifest["layer_catalog"] = layer_specs_to_dicts(layers)
    base_manifest["layer_summary"] = layer_summary

    return base_manifest


def read_sidecar_metadata_if_available(raster_entry: dict[str, Any]) -> dict[str, Any] | None:
    """
    Read copied sidecar metadata JSON if available.
    """
    path_str = raster_entry.get("sidecar_json_dataset_path")
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    return load_yaml(path)