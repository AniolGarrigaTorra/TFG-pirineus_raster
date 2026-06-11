from __future__ import annotations

from pathlib import Path

from src.io.paths import ensure_dir
from src.pipeline.progress import (
    progress_advance_stage_task,
    progress_log,
    progress_set_stage_task_total,
)
from src.sources.copernicus.download import (
    copy_local_file,
    download_file,
    download_multiple_files,
)
from src.sources.copernicus.postprocess import run_static_postprocess
from src.sources.generic_raster.naming import (
    get_download_file_specs,
    validate_generic_raster_source_config,
)


def _run_postprocess(
    *,
    input_paths: list[Path],
    output_path: Path,
    source_cfg: dict,
    spec: dict,
) -> list[Path]:
    written = run_static_postprocess(
        postprocess=spec.get("postprocess"),
        input_paths=input_paths,
        output_path=output_path,
        spec=spec,
        source_cfg=source_cfg,
    )
    return [written]


def download_generic_raster_raw_files(
    source_cfg: dict,
    raw_dir: Path,
    provider: str | None = None,
    required_variables: set[str] | None = None,
) -> list[Path]:
    validate_generic_raster_source_config(source_cfg, provider=provider)

    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)

    download_cfg = source_cfg.get("download", {}) or {}
    enabled = bool(download_cfg.get("enabled", True))
    mode = str(download_cfg.get("mode", "manual")).lower()
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    progress_log(f"[download] Generic raster raw dir: {raw_dir}")
    progress_log(f"[download] Provider: {source_cfg['source']['provider']}")
    progress_log(f"[download] Mode: {mode}")
    progress_log(f"[download] Enabled: {enabled}")

    specs = get_download_file_specs(source_cfg)
    
    # Filter specs by required_variables if provided
    if required_variables:
        specs_before = len(specs)
        # Use prefix matching to handle expanded variable names like agb_2005 matching agb
        filtered_specs = []
        for spec in specs:
            spec_var = spec["variable"]
            # Check if spec variable matches any required variable exactly, or starts with required_var_
            for req_var in required_variables:
                if spec_var == req_var or spec_var.startswith(f"{req_var}_"):
                    filtered_specs.append(spec)
                    break
        specs = filtered_specs
        progress_log(f"[download] Filtered specs from {specs_before} to {len(specs)}")
    
    progress_set_stage_task_total(
        sum(len(spec.get("urls") or []) or 1 for spec in specs),
        label="downloads",
    )

    raw_paths: list[Path] = []

    for spec in specs:
        variable = spec["variable"]
        output_path = raw_dir / spec["filename"]

        progress_log(f"[download] Variable/download spec: {variable}")
        progress_log(f"[download] File: {output_path}")

        if not enabled or mode == "manual":
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Expected raw file does not exist: {output_path}\n"
                    "Place the file there manually, switch to mode=local_file, "
                    "or configure download.files.<name>.url/urls with mode=manual_url."
            )
            progress_log(f"[download] Manual file found: {output_path}")
            progress_advance_stage_task(name=output_path.name)
            raw_paths.append(output_path)
            continue

        if mode == "manual_url":
            url = spec.get("url")
            urls = spec.get("urls")
            if urls:
                downloaded = download_multiple_files(
                    urls=urls,
                    output_dir=raw_dir / "_parts" / variable,
                    filenames=spec.get("filenames"),
                    overwrite=overwrite,
                )
                raw_paths.extend(
                    _run_postprocess(
                        input_paths=downloaded,
                        output_path=output_path,
                        source_cfg=source_cfg,
                        spec=spec,
                    )
                )
                progress_advance_stage_task(name=output_path.name)
                continue
            if url:
                download_file(
                    url=url,
                    output_path=output_path,
                    overwrite=overwrite,
                )
                raw_paths.append(output_path)
                continue
            raise ValueError(
                f"Missing url or urls for variable={variable!r} in download.files"
            )

        if mode == "local_file":
            local_path = spec.get("local_path")
            if not local_path:
                raise ValueError(
                    f"Missing local_path for variable={variable!r} in download.files"
                )
            raw_paths.append(
                copy_local_file(
                    local_path=local_path,
                    output_path=output_path,
                    overwrite=overwrite,
                )
            )
            continue

        raise NotImplementedError(
            f"Unsupported generic raster download mode={mode!r}. "
            "Supported modes: manual, manual_url, local_file"
        )

    return raw_paths
