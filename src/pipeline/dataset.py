from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.io.config import load_yaml, resolve_path
from src.io.paths import get_project_base_dir
from src.pipeline.config import (
    get_dataset_dir,
    get_project_config_path,
    get_run_aoi_config_path,
    get_run_name,
    get_run_resolution_m,
    get_source_entries,
    load_run_config,
    normalize_stages,
)
from src.pipeline.derived import build_derived_features
from src.pipeline.project_overrides import apply_run_overrides_to_project_cfg
from src.pipeline.layers import (
    build_layer_catalog_from_manifest,
    layer_specs_to_dicts,
    summarize_layer_catalog,
)
from src.pipeline.runner import run_source_pipeline
from src.workbench.compiler import compile_run_config


# =============================================================================
# Generic JSON / filesystem helpers
# =============================================================================


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def copy_text_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f_src:
        content = f_src.read()

    with dst.open("w", encoding="utf-8") as f_dst:
        f_dst.write(content)


def path_relative_to_base(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


# =============================================================================
# Dataset directories
# =============================================================================


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


# =============================================================================
# Raster copying into dataset folder
# =============================================================================


def safe_dataset_filename(
    source_id: str,
    source_path: Path,
) -> str:
    """
    Build a stable filename for a copied dataset raster.

    Current feature filenames are already descriptive, so we keep them.
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
        "dataset_path": path_relative_to_base(dst_raster, dataset_rasters_dir.parent),
        "sidecar_json_original_path": str(src_json) if src_json.exists() else None,
        "sidecar_json_dataset_path": path_relative_to_base(
            dst_json,
            dataset_rasters_dir.parent,
        ),
    }


def copy_stage_rasters_to_dataset(
    source_id: str,
    stage_results: list[dict[str, Any]],
    dataset_rasters_dir: Path,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """
    Copy rasters from build stage results to the dataset rasters directory.

    Only .tif/.tiff files from stage='build' are copied.
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


# =============================================================================
# Manifest
# =============================================================================


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
    run_crs: str | None = None,
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
        "run_aoi_config": str(run_aoi_config_path) if run_aoi_config_path else None,
        "run_aoi_name": run_aoi_name,
        "run_resolution_m": run_resolution_m,
        "run_crs": run_crs,
        "n_sources": len(source_results),
        "n_rasters": len(copied_rasters),
        "sources": source_results,
        "rasters": copied_rasters,
    }

    layers = build_layer_catalog_from_manifest(base_manifest)
    layer_summary = summarize_layer_catalog(layers)

    base_manifest["layer_catalog"] = layer_specs_to_dicts(layers)
    base_manifest["layer_summary"] = layer_summary

    return base_manifest


# =============================================================================
# Dataset run orchestration
# =============================================================================


def _load_run_aoi(
    run_cfg: dict[str, Any],
    run_config_path: Path | None = None,
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    run_aoi_config_path = get_run_aoi_config_path(run_cfg)

    if run_aoi_config_path is None:
        return None, None, None

    run_aoi_config_path = resolve_path(
        run_aoi_config_path,
        base_path=run_config_path,
        must_exist=True,
    )
    run_aoi_cfg = load_yaml(run_aoi_config_path)
    run_aoi_name = run_aoi_cfg.get("name") or run_aoi_cfg.get("aoi", {}).get("name")

    return run_aoi_config_path, run_aoi_cfg, run_aoi_name


def run_dataset_pipeline(
    run_config_path: str | Path,
) -> dict[str, Any]:
    """
    Execute a complete dataset run_config.

    Main high-level execution path:
      1. execute all requested sources/stages
      2. copy final rasters into the dataset folder
      3. write run_summary.json
      4. write manifest.json
      5. build derived features, if configured
      6. update manifest.json
    """
    run_config_path = resolve_path(run_config_path, must_exist=True)

    run_cfg = compile_run_config(load_run_config(run_config_path))
    run_name = get_run_name(run_cfg)
    project_config_path = resolve_path(
        get_project_config_path(run_cfg),
        base_path=run_config_path,
        must_exist=True,
    )
    project_cfg = load_yaml(project_config_path)
    project_cfg["_config_path"] = str(project_config_path)
    project_cfg = apply_run_overrides_to_project_cfg(project_cfg, run_cfg)
    dataset_dir = resolve_path(
        get_dataset_dir(run_cfg),
        base_path=get_project_base_dir(project_cfg),
    )

    run_aoi_config_path, run_aoi_cfg, run_aoi_name = _load_run_aoi(
        run_cfg,
        run_config_path=run_config_path,
    )
    run_resolution_m = get_run_resolution_m(run_cfg)

    run_cfg.setdefault("run", {})["project_config"] = str(project_config_path)
    if run_aoi_config_path is not None:
        run_cfg["run"]["aoi_config"] = str(run_aoi_config_path)
    if run_cfg["run"].get("clip_aoi_config"):
        run_cfg["run"]["clip_aoi_config"] = str(
            resolve_path(
                run_cfg["run"]["clip_aoi_config"],
                base_path=run_config_path,
                must_exist=True,
            )
        )

    outputs_cfg = run_cfg.get("outputs", {})
    copy_rasters = bool(outputs_cfg.get("copy_rasters", True))
    overwrite_existing = bool(outputs_cfg.get("overwrite_existing", True))
    write_run_summary = bool(outputs_cfg.get("write_run_summary", True))
    write_manifest = bool(outputs_cfg.get("write_manifest", True))

    dirs = ensure_dataset_dirs(
        dataset_dir=dataset_dir,
        project_cfg=project_cfg,
    )

    copy_text_file(
        run_config_path,
        dirs["config"] / "run_config.yaml",
    )
    copy_text_file(
        project_config_path,
        dirs["config"] / "project_config.yaml",
    )

    if run_aoi_config_path is not None:
        resolved_run_aoi_config_path = resolve_path(
            run_aoi_config_path,
            base_path=run_config_path,
            must_exist=True,
        )
        copy_text_file(
            resolved_run_aoi_config_path,
            dirs["config"] / "aoi_config.yaml",
        )

    print("==============================")
    print("Pirineus Raster Dataset Run")
    print(f"Run name:       {run_name}")
    print(f"Run config:     {run_config_path}")
    print(f"Project config: {project_config_path}")
    print(f"Dataset dir:    {dataset_dir}")
    print(f"Copy rasters:   {copy_rasters}")
    print("==============================")

    started_at = now_iso()

    source_entries = get_source_entries(run_cfg)
    default_stages = normalize_stages(run_cfg["run"].get("stages", ["build"]))

    source_results: list[dict[str, Any]] = []
    copied_rasters: list[dict[str, Any]] = []

    for idx, source_entry in enumerate(source_entries, start=1):
        source_config_path = resolve_path(
            source_entry["config"],
            base_path=run_config_path,
            must_exist=True,
        )
        source_id = source_entry.get("id", source_config_path.stem)
        stages = normalize_stages(source_entry.get("stages", default_stages))

        copy_text_file(
            source_config_path,
            dirs["config"] / "sources" / f"{source_id}.yaml",
        )

        print("==============================")
        print(f"Source {idx}/{len(source_entries)}: {source_id}")
        print(f"Config: {source_config_path}")
        print(f"Stages: {stages}")
        print("==============================")

        stage_results: list[dict[str, Any]] = []

        for stage in stages:
            print("------------------------------")
            print(f"Running source '{source_id}' stage '{stage}'")
            print("------------------------------")

            paths = run_source_pipeline(
                project_config_path=project_config_path,
                source_config_path=source_config_path,
                stage=stage,
                run_cfg=run_cfg,
                source_entry=source_entry,
            )

            stage_results.append(
                {
                    "stage": stage,
                    "n_paths": len(paths),
                    "paths": [str(path) for path in paths],
                }
            )

        source_result = {
            "id": source_id,
            "config": str(source_config_path),
            "stages": stages,
            "stage_results": stage_results,
        }

        if copy_rasters:
            source_copied = copy_stage_rasters_to_dataset(
                source_id=source_id,
                stage_results=stage_results,
                dataset_rasters_dir=dirs["rasters"],
                overwrite=overwrite_existing,
            )

            source_result["copied_rasters"] = source_copied
            source_result["n_copied_rasters"] = len(source_copied)

            copied_rasters.extend(source_copied)

            print("------------------------------")
            print(f"Copied rasters for source '{source_id}': {len(source_copied)}")
            print("------------------------------")

        source_results.append(source_result)

    finished_at = now_iso()

    summary: dict[str, Any] = {
        "run_name": run_name,
        "run_config": str(run_config_path),
        "project_config": str(project_config_path),
        "dataset_dir": str(dataset_dir),
        "run_aoi_config": str(run_aoi_config_path) if run_aoi_config_path else None,
        "run_aoi_name": run_aoi_name,
        "run_resolution_m": run_resolution_m,
        "run_crs": project_cfg.get("crs"),
        "started_at": started_at,
        "finished_at": finished_at,
        "copy_rasters": copy_rasters,
        "n_sources": len(source_results),
        "n_copied_rasters": len(copied_rasters),
        "sources": source_results,
    }

    if write_run_summary:
        write_json(
            dirs["metadata"] / "run_summary.json",
            summary,
        )

    if write_manifest:
        manifest = build_manifest(
            run_name=run_name,
            run_config_path=run_config_path,
            project_config_path=project_config_path,
            dataset_dir=dataset_dir,
            source_results=source_results,
            copied_rasters=copied_rasters,
            started_at=started_at,
            finished_at=finished_at,
            run_aoi_config_path=run_aoi_config_path,
            run_aoi_name=run_aoi_name,
            run_resolution_m=run_resolution_m,
            run_crs=project_cfg.get("crs"),
        )

        write_json(
            dirs["metadata"] / "manifest.json",
            manifest,
        )

    derived_paths: list[Path] = []

    if run_cfg.get("derived_features"):
        if run_aoi_cfg is None:
            raise ValueError(
                "derived_features require run.aoi_config to be defined."
            )

        derived_paths = build_derived_features(
            run_cfg=run_cfg,
            project_cfg=project_cfg,
            dataset_dir=dataset_dir,
            output_aoi_cfg=run_aoi_cfg,
        )

        summary["n_derived_features"] = len(derived_paths)
        summary["derived_features"] = [str(path) for path in derived_paths]

        if write_run_summary:
            write_json(
                dirs["metadata"] / "run_summary.json",
                summary,
            )

    print("==============================")
    print("Dataset run finished successfully")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Copied rasters: {len(copied_rasters)}")
    print(f"Derived rasters: {len(derived_paths)}")
    print(f"Run summary: {dirs['metadata'] / 'run_summary.json'}")
    print(f"Manifest:    {dirs['metadata'] / 'manifest.json'}")
    print("==============================")

    return summary
