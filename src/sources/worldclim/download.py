from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import shutil
import time

from src.io.paths import ensure_dir
from src.sources.worldclim.naming import (
    build_worldclim_download_url,
    build_worldclim_zip_path,
    get_zip_variable_codes,
)


USER_AGENT = "pirineus-raster-pipeline/0.1"


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
        print(f"[download] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    if temporary_path.exists() and not overwrite:
        print(f"[download] Partial file exists, removing: {temporary_path}")
        temporary_path.unlink()

    print(f"[download] URL: {url}")
    print(f"[download] Output: {output_path}")

    last_error = None

    for attempt in range(1, max_retries + 1):
        print(f"[download] Attempt {attempt}/{max_retries}")

        request = Request(
            url,
            headers={"User-Agent": USER_AGENT},
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                with temporary_path.open("wb") as f:
                    shutil.copyfileobj(response, f)

            temporary_path.rename(output_path)
            print(f"[download] Finished: {output_path}")
            return

        except (HTTPError, URLError, TimeoutError) as e:
            last_error = e

            if temporary_path.exists():
                temporary_path.unlink()

            print(f"[download] Failed attempt {attempt}/{max_retries}: {e}")

            if attempt < max_retries:
                print(
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


def ensure_worldclim_zip(
    source_cfg: dict,
    raw_dir: Path,
    zip_variable_code: str,
) -> Path:
    download_cfg = source_cfg.get("download", {})

    mode = download_cfg.get("mode", "manual")
    enabled = bool(download_cfg.get("enabled", False))
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    zip_path = build_worldclim_zip_path(
        raw_dir=raw_dir,
        source_cfg=source_cfg,
        zip_variable_code=zip_variable_code,
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
        source_cfg=source_cfg,
        zip_variable_code=zip_variable_code,
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
) -> list[Path]:
    ensure_dir(raw_dir)

    zip_variable_codes = get_zip_variable_codes(source_cfg)
    zip_paths = []

    print("[worldclim] ZIP variable codes:", ", ".join(zip_variable_codes))

    for zip_variable_code in zip_variable_codes:
        zip_path = ensure_worldclim_zip(
            source_cfg=source_cfg,
            raw_dir=raw_dir,
            zip_variable_code=zip_variable_code,
        )
        zip_paths.append(zip_path)

    return zip_paths