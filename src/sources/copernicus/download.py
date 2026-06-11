from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import shutil
import time

from src.io.paths import ensure_dir
from src.pipeline.progress import (
    progress_advance_stage_task,
    progress_download,
    progress_log,
    progress_set_stage_task_total,
)
from src.sources.copernicus.hda import download_with_wekeo_hda
from src.sources.copernicus.naming import (
    validate_copernicus_source_config,
    get_download_file_specs,
)
from src.sources.copernicus.postprocess import run_static_postprocess
from src.sources.copernicus.temporal_postprocess import (
    run_temporal_zip_geotiff_aggregation,
)

USER_AGENT = "pirineus-raster-pipeline/0.1"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name

    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")

    return name


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    timeout: int = 900,
    max_retries: int = 3,
    retry_sleep_seconds: int = 30,
) -> None:
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        progress_log(f"[download] Exists, skipping: {output_path}")
        progress_advance_stage_task(name=output_path.name)
        return

    ensure_dir(output_path.parent)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    if temporary_path.exists():
        progress_log(f"[download] Removing partial file: {temporary_path}")
        temporary_path.unlink()

    progress_log(f"[download] URL: {url}")
    progress_log(f"[download] Output: {output_path}")

    last_error = None

    for attempt in range(1, max_retries + 1):
        progress_log(f"[download] Attempt {attempt}/{max_retries}")

        request = Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                with temporary_path.open("wb") as f:
                    total_header = response.headers.get("Content-Length")
                    total = int(total_header) if total_header else None
                    downloaded = 0
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_download(
                            output_path=output_path,
                            downloaded=downloaded,
                            total=total,
                            attempt=attempt,
                        )

            temporary_path.rename(output_path)
            progress_download(
                output_path=output_path,
                downloaded=output_path.stat().st_size,
                total=output_path.stat().st_size,
                attempt=attempt,
                done=True,
            )
            progress_log(f"[download] Finished: {output_path}")
            progress_advance_stage_task(name=output_path.name)
            return

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if temporary_path.exists():
                temporary_path.unlink()

            progress_log(f"[download] Failed attempt {attempt}/{max_retries}: {exc}", level="warning")

            if attempt < max_retries:
                progress_log(
                    f"[download] Sleeping {retry_sleep_seconds} seconds before retry..."
                )
                time.sleep(retry_sleep_seconds)

        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise

    raise RuntimeError(
        "Failed to download Copernicus file after multiple attempts.\n"
        f"URL: {url}\n"
        f"Output: {output_path}\n"
        f"Last error: {last_error}"
    )


def download_multiple_files(
    urls: list[str],
    output_dir: Path,
    filenames: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if filenames is not None and len(filenames) != len(urls):
        raise ValueError(
            "If filenames is provided, it must have the same length as urls."
        )

    downloaded: list[Path] = []

    for idx, url in enumerate(urls):
        filename = filenames[idx] if filenames is not None else _filename_from_url(url)
        output_path = output_dir / filename

        download_file(
            url=url,
            output_path=output_path,
            overwrite=overwrite,
        )

        downloaded.append(output_path)

    return downloaded


def copy_local_file(
    local_path: str | Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    local_path = Path(local_path)
    output_path = Path(output_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local source file not found: {local_path}")

    if output_path.exists() and not overwrite:
        progress_log(f"[download] Exists, skipping: {output_path}")
        progress_advance_stage_task(name=output_path.name)
        return output_path

    ensure_dir(output_path.parent)

    progress_log(f"[download] Copy local file: {local_path}")
    progress_log(f"[download] Output: {output_path}")

    shutil.copy2(local_path, output_path)
    progress_advance_stage_task(name=output_path.name)
    return output_path


def _run_postprocess(
    *,
    input_paths: list[Path],
    output_path: Path,
    raw_dir: Path,
    source_cfg: dict,
    spec: dict,
) -> list[Path]:
    postprocess = spec.get("postprocess") or "copy_single"

    if postprocess == "temporal_zip_geotiff_aggregation":
        return run_temporal_zip_geotiff_aggregation(
            input_paths=input_paths,
            raw_dir=raw_dir,
            source_cfg=source_cfg,
            spec=spec,
        )

    written = run_static_postprocess(
        postprocess=postprocess,
        input_paths=input_paths,
        output_path=output_path,
        spec=spec,
        source_cfg=source_cfg,
    )

    return [written]


def _handle_manual_url_download(
    *,
    spec: dict,
    output_path: Path,
    raw_dir: Path,
    overwrite: bool,
    source_cfg: dict,
) -> list[Path]:
    variable = spec["variable"]
    url = spec.get("url")
    urls = spec.get("urls")

    if urls:
        if not isinstance(urls, list):
            raise TypeError(
                f"download.files.{variable}.urls must be a list of URLs."
            )

        parts_dir = raw_dir / "_parts" / variable

        downloaded_paths = download_multiple_files(
            urls=urls,
            output_dir=parts_dir,
            filenames=spec.get("filenames"),
            overwrite=overwrite,
        )

        return _run_postprocess(
            input_paths=downloaded_paths,
            output_path=output_path,
            raw_dir=raw_dir,
            source_cfg=source_cfg,
            spec=spec,
        )

    if url:
        download_file(
            url=url,
            output_path=output_path,
            overwrite=overwrite,
        )
        return [output_path]

    raise ValueError(
        f"Missing url or urls for variable={variable!r} in download.files"
    )


def _handle_local_file_download(
    *,
    spec: dict,
    output_path: Path,
    overwrite: bool,
) -> list[Path]:
    local_path = spec.get("local_path")
    if not local_path:
        raise ValueError(
            f"Missing local_path for variable={spec['variable']!r} in download.files"
        )

    copied = copy_local_file(
        local_path=local_path,
        output_path=output_path,
        overwrite=overwrite,
    )

    return [copied]


def download_copernicus_raw_files(
    source_cfg: dict,
    raw_dir: Path,
    required_variables: set[str] | None = None,
) -> list[Path]:
    validate_copernicus_source_config(source_cfg)

    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)

    download_cfg = source_cfg.get("download", {}) or {}

    enabled = bool(download_cfg.get("enabled", True))
    mode = str(download_cfg.get("mode", "manual")).lower()
    overwrite = bool(download_cfg.get("overwrite_existing", False))

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

    progress_log(f"[download] Copernicus raw dir: {raw_dir}")
    progress_log(f"[download] Mode: {mode}")
    progress_log(f"[download] Enabled: {enabled}")

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
                    "Either place the file manually there, or configure an "
                    "automatic download mode in the YAML."
                )

            progress_log(f"[download] Manual file found: {output_path}")
            progress_advance_stage_task(name=output_path.name)
            raw_paths.append(output_path)
            continue

        if mode == "manual_url":
            written_paths = _handle_manual_url_download(
                spec=spec,
                output_path=output_path,
                raw_dir=raw_dir,
                overwrite=overwrite,
                source_cfg=source_cfg,
            )
            raw_paths.extend(written_paths)
            progress_advance_stage_task(name=output_path.name)
            continue

        if mode == "local_file":
            written_paths = _handle_local_file_download(
                spec=spec,
                output_path=output_path,
                overwrite=overwrite,
            )
            raw_paths.extend(written_paths)
            progress_advance_stage_task(name=output_path.name)
            continue

        if mode == "wekeo_hda":
            downloaded_files = download_with_wekeo_hda(
                source_cfg=source_cfg,
                spec=spec,
                output_path=output_path,
            )

            # If empty list returned, it means cache hit - output file already exists
            if not downloaded_files:
                if output_path.exists():
                    progress_log(f"[download] ✓ Cached: {output_path}")
                    raw_paths.append(output_path)
                else:
                    raise FileNotFoundError(
                        f"Cache hit detected but output file not found: {output_path}"
                    )
            else:
                written_paths = _run_postprocess(
                    input_paths=downloaded_files,
                    output_path=output_path,
                    raw_dir=raw_dir,
                    source_cfg=source_cfg,
                    spec=spec,
                )
                raw_paths.extend(written_paths)
            
            progress_advance_stage_task(name=output_path.name)
            continue

        raise NotImplementedError(
            f"Unsupported Copernicus download mode={mode!r}. "
            "Supported modes: manual, manual_url, local_file, wekeo_hda"
        )

    return raw_paths
