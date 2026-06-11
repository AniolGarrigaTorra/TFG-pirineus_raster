from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import time

from src.io.paths import ensure_dir
from src.pipeline.progress import (
    progress_advance_stage_task,
    progress_download,
    progress_log,
    progress_set_stage_task_total,
)
from src.sources.worldclim.naming import (
    build_worldclim_download_url,
    build_worldclim_zip_path,
    get_zip_specs,
    get_file_specs,
    get_layer_structure,
    build_worldclim_cmip6_download_url,
    build_worldclim_cmip6_raw_path,
)
USER_AGENT = "pirineus-raster-pipeline/0.1"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    timeout: int = 600,
    max_retries: int = 3,
    retry_sleep_seconds: int = 30,
) -> None:
    """
    Download a file using only the Python standard library.

    Includes retries because WorldClim downloads from HPC environments may fail
    during connection setup or HTTPS handshake.
    """
    if output_path.exists() and not overwrite:
        progress_log(f"[download] Exists, skipping: {output_path}")
        progress_advance_stage_task(name=output_path.name)
        return

    ensure_dir(output_path.parent)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    if temporary_path.exists() and not overwrite:
        progress_log(f"[download] Partial file exists, removing: {temporary_path}")
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

        except (HTTPError, URLError, TimeoutError) as e:
            last_error = e

            if temporary_path.exists():
                temporary_path.unlink()

            progress_log(f"[download] Failed attempt {attempt}/{max_retries}: {e}", level="warning")

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
        "Failed to download after multiple attempts.\n"
        f"URL: {url}\n"
        f"Output: {output_path}\n"
        f"Last error: {last_error}\n\n"
        "Manual fallback:\n"
        f"  mkdir -p {output_path.parent}\n"
        f"  wget -c -O {output_path} {url}\n"
        "Then re-run the pipeline."
    )


def download_worldclim_direct_files(
    source_cfg: dict,
    raw_dir: Path,
) -> list[Path]:
    ensure_dir(raw_dir)

    file_specs = get_file_specs(source_cfg)
    progress_set_stage_task_total(len(file_specs), label="downloads")
    download_cfg = source_cfg.get("download", {})

    mode = download_cfg.get("mode", "manual")
    enabled = bool(download_cfg.get("enabled", False))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    paths: list[Path] = []

    progress_log("[worldclim] Direct file specs:")
    for spec in file_specs:
        progress_log(
            f"  - {spec['variable']} "
            f"{spec['gcm']} {spec['ssp']} {spec['period']}"
        )

    for spec in file_specs:
        output_path = build_worldclim_cmip6_raw_path(
            raw_dir=raw_dir,
            source_cfg=source_cfg,
            file_spec=spec,
        )

        url = build_worldclim_cmip6_download_url(
            source_cfg=source_cfg,
            file_spec=spec,
        )

        if output_path.exists() and not overwrite:
            progress_log(f"[worldclim] Raw file already exists: {output_path}")
            progress_advance_stage_task(name=output_path.name)
            paths.append(output_path)
            continue

        if not enabled or mode == "manual":
            raise FileNotFoundError(
                "WorldClim CMIP6 raw GeoTIFF not found and automatic download is disabled.\n"
                f"Expected file: {output_path}\n"
                "Manual protocol:\n"
                f"  mkdir -p {output_path.parent}\n"
                f"  wget -c -O {output_path} {url}\n"
                "  Re-run the pipeline."
            )

        if mode != "auto":
            raise ValueError(f"Unsupported download mode: {mode}. Use 'auto' or 'manual'.")

        download_file(
            url=url,
            output_path=output_path,
            overwrite=overwrite,
        )

        time.sleep(1)
        paths.append(output_path)

    return paths


def ensure_worldclim_zip(
    source_cfg: dict,
    raw_dir: Path,
    zip_spec: dict,
) -> Path:
    download_cfg = source_cfg.get("download", {})

    mode = download_cfg.get("mode", "manual")
    enabled = bool(download_cfg.get("enabled", False))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    zip_path = build_worldclim_zip_path(
        raw_dir=raw_dir,
        source_cfg=source_cfg,
        zip_spec=zip_spec,
    )

    if zip_path.exists() and not overwrite:
        progress_log(f"[worldclim] Raw ZIP already exists: {zip_path}")
        progress_advance_stage_task(name=zip_path.name)
        return zip_path

    if not enabled or mode == "manual":
        if not zip_path.exists():
            url = build_worldclim_download_url(
                source_cfg=source_cfg,
                zip_spec=zip_spec,
            )
            raise FileNotFoundError(
                "WorldClim raw ZIP not found and automatic download is disabled.\n"
                f"Expected file: {zip_path}\n"
                "Manual protocol:\n"
                f"  mkdir -p {zip_path.parent}\n"
                f"  wget -c -O {zip_path} {url}\n"
                "  Re-run the pipeline."
            )

        progress_log(f"[worldclim] Manual mode. Found existing ZIP: {zip_path}")
        progress_advance_stage_task(name=zip_path.name)
        return zip_path

    if mode != "auto":
        raise ValueError(f"Unsupported download mode: {mode}. Use 'auto' or 'manual'.")

    url = build_worldclim_download_url(
        source_cfg=source_cfg,
        zip_spec=zip_spec,
    )

    download_file(
        url=url,
        output_path=zip_path,
        overwrite=overwrite,
    )

    time.sleep(1)

    return zip_path


def download_worldclim_raw_files(
    source_cfg: dict,
    raw_dir: Path,
    required_variables: set[str] | None = None,
) -> list[Path]:
    layer_structure = get_layer_structure(source_cfg)

    if layer_structure == "future_monthly_multiband":
        return download_worldclim_direct_files(
            source_cfg=source_cfg,
            raw_dir=raw_dir,
        )

    ensure_dir(raw_dir)

    zip_specs = get_zip_specs(source_cfg)
    progress_set_stage_task_total(len(zip_specs), label="downloads")
    zip_paths = []

    progress_log("[worldclim] ZIP specs:")
    for spec in zip_specs:
        label = spec["zip_variable_code"]
        if spec.get("period"):
            label = f"{label}_{spec['period']}"
        progress_log(f"  - {label}")

    for zip_spec in zip_specs:
        zip_path = ensure_worldclim_zip(
            source_cfg=source_cfg,
            raw_dir=raw_dir,
            zip_spec=zip_spec,
        )
        zip_paths.append(zip_path)

    return zip_paths
