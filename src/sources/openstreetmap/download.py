from __future__ import annotations

from pathlib import Path

from src.io.paths import ensure_dir
from src.pipeline.progress import (
    progress_advance_stage_task,
    progress_log,
    progress_set_stage_task_total,
)
from src.sources.copernicus.download import download_file
from src.sources.openstreetmap.naming import build_osm_raw_path, validate_osm_source_config


def download_osm_raw_files(source_cfg: dict, raw_dir: Path, required_variables: set[str] | None = None) -> list[Path]:
    validate_osm_source_config(source_cfg)

    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)

    download_cfg = source_cfg.get("download", {}) or {}
    enabled = bool(download_cfg.get("enabled", True))
    mode = str(download_cfg.get("mode", "manual_url")).lower()
    overwrite = bool(download_cfg.get("overwrite_existing", False))
    regions = download_cfg.get("regions", []) or []

    if not regions:
        raise ValueError("OpenStreetMap source requires download.regions.")

    progress_log(f"[download:osm] Raw dir: {raw_dir}")
    progress_log(f"[download:osm] Mode: {mode}")
    progress_log(f"[download:osm] Regions: {len(regions)}")
    progress_set_stage_task_total(len(regions), label="downloads")

    raw_paths: list[Path] = []

    for region_cfg in regions:
        name = str(region_cfg["name"])
        output_path = build_osm_raw_path(raw_dir, region_cfg)
        url = region_cfg.get("url")

        progress_log(f"[download:osm] Region: {name}")
        progress_log(f"[download:osm] File: {output_path}")

        if not enabled or mode == "manual":
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Expected OSM PBF does not exist: {output_path}\n"
                    "Place the Geofabrik .osm.pbf file there manually, or use "
                    "download.mode=manual_url with region URLs."
                )
            progress_advance_stage_task(name=output_path.name)
            raw_paths.append(output_path)
            continue

        if mode == "manual_url":
            if not url:
                raise ValueError(f"Missing url for OSM region {name!r}.")
            download_file(
                url=str(url),
                output_path=output_path,
                overwrite=overwrite,
                timeout=int(download_cfg.get("timeout_seconds", 1800)),
                max_retries=int(download_cfg.get("max_retries", 3)),
                retry_sleep_seconds=int(download_cfg.get("retry_sleep_seconds", 30)),
            )
            raw_paths.append(output_path)
            continue

        raise NotImplementedError(
            f"Unsupported OpenStreetMap download mode={mode!r}. "
            "Supported modes: manual, manual_url"
        )

    return raw_paths
