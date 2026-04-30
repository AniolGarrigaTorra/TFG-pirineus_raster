from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io.config import load_yaml
from src.pipeline.dataset_output import (
    build_manifest,
    copy_stage_rasters_to_dataset,
    copy_text_file,
    ensure_dataset_dirs,
    now_iso,
    write_json,
)
from src.pipeline.run_config import (
    get_dataset_dir,
    get_project_config_path,
    get_run_aoi_config_path,
    get_run_name,
    get_run_resolution_m,
    get_source_entries,
    load_run_config,
    normalize_stages,
)
from src.pipeline.runner import run_source_pipeline


def run_dataset_pipeline(
    run_config_path: str | Path,
) -> dict[str, Any]:
    """
    Execute a complete dataset run_config.

    This is now the main high-level execution path of the repository.
    It:
    - loads one run_config
    - executes all requested sources/stages
    - copies final rasters into the dataset folder
    - writes run_summary.json
    - writes manifest.json
    """
    run_config_path = Path(run_config_path)

    run_cfg = load_run_config(run_config_path)
    run_name = get_run_name(run_cfg)
    project_config_path = get_project_config_path(run_cfg)
    project_cfg = load_yaml(project_config_path)
    dataset_dir = get_dataset_dir(run_cfg)

    run_aoi_config_path = get_run_aoi_config_path(run_cfg)
    run_resolution_m = get_run_resolution_m(run_cfg)

    run_aoi_name = None
    if run_aoi_config_path is not None:
        run_aoi_cfg = load_yaml(run_aoi_config_path)
        run_aoi_name = run_aoi_cfg.get("name") or run_aoi_cfg.get("aoi", {}).get("name")

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
        source_config_path = Path(source_entry["config"])
        source_id = source_entry.get("id", source_config_path.stem)
        stages = normalize_stages(source_entry.get("stages", default_stages))

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
        "started_at": started_at,
        "finished_at": finished_at,
        "copy_rasters": copy_rasters,
        "n_sources": len(source_results),
        "n_copied_rasters": len(copied_rasters),
        "sources": source_results,
        "run_aoi_config": str(run_aoi_config_path) if run_aoi_config_path else None,
        "run_aoi_name": run_aoi_name,
        "run_resolution_m": run_resolution_m,
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
        )

        write_json(
            dirs["metadata"] / "manifest.json",
            manifest,
        )

    print("==============================")
    print("Dataset run finished successfully")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Copied rasters: {len(copied_rasters)}")
    print(f"Run summary: {dirs['metadata'] / 'run_summary.json'}")
    print(f"Manifest:    {dirs['metadata'] / 'manifest.json'}")
    print("==============================")

    return summary