from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import shutil
import time

from src.io.paths import ensure_dir
from src.sources.worldclim.naming import (
    build_worldclim_download_url,
    build_worldclim_zip_path,
)


USER_AGENT = "pirineus-raster-pipeline/0.1"


def get_enabled_variables(source_cfg: dict) -> list[str]:
    variables_cfg = source_cfg.get("variables", {})
    enabled = []

    for variable, cfg in variables_cfg.items():
        if cfg.get("enabled", False):
            enabled.append(variable)

    if not enabled:
        raise ValueError("No enabled variables found in source config.")

    return enabled


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    timeout: int = 120,
) -> None:
    """
    Download a file using only the Python standard library.

    This avoids adding extra dependencies such as requests.
    """
    if output_path.exists() and not overwrite:
        print(f"[download] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    if temporary_path.exists():
        temporary_path.unlink()

    print(f"[download] URL: {url}")
    print(f"[download] Output: {output_path}")

    request = Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            with temporary_path.open("wb") as f:
                shutil.copyfileobj(response, f)

        temporary_path.rename(output_path)

    except HTTPError as e:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"HTTP error while downloading {url}: {e}") from e

    except URLError as e:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(f"URL error while downloading {url}: {e}") from e

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(f"[download] Finished: {output_path}")


def ensure_worldclim_zip(
    source_cfg: dict,
    raw_dir: Path,
    variable: str,
) -> Path:
    source = source_cfg["source"]
    download_cfg = source_cfg.get("download", {})
    processing_cfg = source_cfg["processing"]

    base_url = source["base_url"]
    source_resolution = processing_cfg["source_resolution"]

    mode = download_cfg.get("mode", "manual")
    enabled = bool(download_cfg.get("enabled", False))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    zip_path = build_worldclim_zip_path(
        raw_dir=raw_dir,
        source_resolution=source_resolution,
        variable=variable,
    )

    if zip_path.exists() and not overwrite:
        print(f"[worldclim] Raw ZIP already exists: {zip_path}")
        return zip_path

    if not enabled or mode == "manual":
        if not zip_path.exists():
            raise FileNotFoundError(
                "WorldClim raw ZIP not found and automatic download is disabled.\n"
                f"Expected file: {zip_path}\n"
                "Manual protocol:\n"
                "  1. Download the ZIP from WorldClim.\n"
                f"  2. Place it at: {zip_path}\n"
                "  3. Re-run the pipeline."
            )

        print(f"[worldclim] Manual mode. Found existing ZIP: {zip_path}")
        return zip_path

    if mode != "auto":
        raise ValueError(f"Unsupported download mode: {mode}. Use 'auto' or 'manual'.")

    url = build_worldclim_download_url(
        base_url=base_url,
        source_resolution=source_resolution,
        variable=variable,
    )

    download_file(
        url=url,
        output_path=zip_path,
        overwrite=overwrite,
    )

    # Be polite with the remote server when downloading multiple files.
    time.sleep(1)

    return zip_path


def download_worldclim_raw_files(
    source_cfg: dict,
    raw_dir: Path,
) -> list[Path]:
    """
    Ensure all enabled WorldClim raw ZIP files exist locally.

    Returns a list of ZIP paths.
    """
    ensure_dir(raw_dir)

    enabled_variables = get_enabled_variables(source_cfg)
    zip_paths = []

    print("[worldclim] Enabled variables:", ", ".join(enabled_variables))

    for variable in enabled_variables:
        zip_path = ensure_worldclim_zip(
            source_cfg=source_cfg,
            raw_dir=raw_dir,
            variable=variable,
        )
        zip_paths.append(zip_path)

    return zip_paths