from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil
import time

from src.io.paths import ensure_dir
from src.sources.copernicus.hda import download_with_wekeo_hda
from src.sources.copernicus.naming import (
    validate_copernicus_source_config,
    get_download_file_specs,
)

USER_AGENT = "pirineus-raster-pipeline/0.1"


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
        print(f"[download] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    if temporary_path.exists():
        print(f"[download] Removing partial file: {temporary_path}")
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

        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

            if temporary_path.exists():
                temporary_path.unlink()

            print(f"[download] Failed attempt {attempt}/{max_retries}: {exc}")

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
        "Failed to download Copernicus file after multiple attempts.\n"
        f"URL: {url}\n"
        f"Output: {output_path}\n"
        f"Last error: {last_error}\n\n"
        "Manual fallback:\n"
        f"  mkdir -p {output_path.parent}\n"
        f"  wget -c -O {output_path} '{url}'\n"
        "Then re-run the pipeline."
    )


def copy_local_file(
    local_path: str | Path,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    local_path = Path(local_path)
    output_path = Path(output_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local source file not found: {local_path}")

    if output_path.exists() and not overwrite:
        print(f"[download] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)

    print(f"[download] Copy local file: {local_path}")
    print(f"[download] Output: {output_path}")

    shutil.copy2(local_path, output_path)


def download_copernicus_raw_files(
    source_cfg: dict,
    raw_dir: Path,
) -> list[Path]:
    """
    Download or validate raw Copernicus files.

    Supported modes:
      - manual: check that files already exist in raw_dir
      - manual_url: download from direct URLs declared in YAML
      - local_file: copy files from local paths declared in YAML
      - wekeo_hda: search and download through WEkEO HDA API
    """
    validate_copernicus_source_config(source_cfg)

    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)

    download_cfg = source_cfg.get("download", {}) or {}

    enabled = bool(download_cfg.get("enabled", True))
    mode = str(download_cfg.get("mode", "manual")).lower()
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    specs = get_download_file_specs(source_cfg)

    print("[download] Copernicus raw dir:", raw_dir)
    print("[download] Mode:", mode)
    print("[download] Enabled:", enabled)

    raw_paths: list[Path] = []

    for spec in specs:
        variable = spec["variable"]
        output_path = raw_dir / spec["filename"]

        print("==============================")
        print(f"[download] Variable: {variable}")
        print(f"[download] File: {output_path}")

        if not enabled or mode == "manual":
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Expected raw file does not exist: {output_path}\n"
                    "Either place the file manually there, or configure an "
                    "automatic download mode in the YAML."
                )

            print(f"[download] Manual file found: {output_path}")
            raw_paths.append(output_path)
            continue

        if mode == "manual_url":
            url = spec.get("url")
            if not url:
                raise ValueError(
                    f"Missing URL for variable={variable!r} in download.files"
                )

            download_file(
                url=url,
                output_path=output_path,
                overwrite=overwrite,
            )
            raw_paths.append(output_path)
            continue

        if mode == "local_file":
            local_path = spec.get("local_path")
            if not local_path:
                raise ValueError(
                    f"Missing local_path for variable={variable!r} in download.files"
                )

            copy_local_file(
                local_path=local_path,
                output_path=output_path,
                overwrite=overwrite,
            )
            raw_paths.append(output_path)
            continue

        if mode == "wekeo_hda":
            downloaded_path = download_with_wekeo_hda(
                source_cfg=source_cfg,
                spec=spec,
                output_path=output_path,
            )
            raw_paths.append(downloaded_path)
            continue

        raise NotImplementedError(
            f"Unsupported Copernicus download mode={mode!r}. "
            "Supported modes: manual, manual_url, local_file, wekeo_hda"
        )

    return raw_paths