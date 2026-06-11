from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from src.io.paths import ensure_dir
from src.pipeline.progress import (
    progress_advance_stage_task,
    progress_download,
    progress_log,
    progress_set_stage_task_total,
)
from src.sources.igme_brgm.naming import build_igme_brgm_zip_name, safe_name


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _iter_enabled_datasets(source_cfg: dict):
    datasets = source_cfg.get("datasets", {})

    for dataset_name, dataset_cfg in datasets.items():
        if dataset_cfg.get("enabled", True):
            yield dataset_name, dataset_cfg


def _download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        progress_log(f"[download] Exists, skipping: {output_path}")
        progress_advance_stage_task(name=output_path.name)
        return output_path

    ensure_dir(output_path.parent)

    progress_log("[download] Downloading IGME/BRGM file")
    progress_log(f"  URL: {url}")
    progress_log(f"  Out: {output_path}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                progress_download(
                    output_path=output_path,
                    downloaded=downloaded,
                    total=total,
                )

        tmp_path.replace(output_path)
        progress_download(
            output_path=output_path,
            downloaded=output_path.stat().st_size,
            total=output_path.stat().st_size,
            done=True,
        )
        progress_advance_stage_task(name=output_path.name)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return output_path


def _guess_zip_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name

    if not name.lower().endswith(".zip"):
        raise ValueError(
            f"Cannot infer ZIP filename from URL: {url}. "
            "Set 'zip_filename' in the source YAML."
        )

    return name


def _extract_zip(
    zip_path: Path,
    extract_dir: Path,
    overwrite: bool = False,
) -> Path:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        progress_log(f"[download] Extracted directory exists, skipping: {extract_dir}")
        return extract_dir

    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)

    ensure_dir(extract_dir)

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Downloaded file is not a valid ZIP: {zip_path}")

    progress_log("[download] Extracting ZIP")
    progress_log(f"  ZIP: {zip_path}")
    progress_log(f"  Dir: {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def download_igme_brgm_raw_files(
    source_cfg: dict,
    raw_dir: Path,
    required_variables: set[str] | None = None,
) -> list[Path]:
    """
    Download and optionally extract IGME/BRGM ZIP files.

    The source config can define several datasets, for example:
    - geology
    - quaternary
    """
    download_cfg = source_cfg.get("download", {})
    overwrite = bool(download_cfg.get("overwrite_existing", False))
    extract_after_download = bool(download_cfg.get("extract_after_download", True))

    ensure_dir(raw_dir)

    progress_log(f"[download] IGME/BRGM raw dir: {raw_dir}")

    output_paths: list[Path] = []
    enabled_datasets = list(_iter_enabled_datasets(source_cfg))
    progress_set_stage_task_total(len(enabled_datasets), label="downloads")

    for dataset_name, dataset_cfg in enabled_datasets:
        url = dataset_cfg.get("url")

        if not url:
            raise ValueError(
                f"Dataset '{dataset_name}' has no URL. "
                "Set datasets.<name>.url in the source YAML."
            )

        zip_filename = (
            dataset_cfg.get("zip_filename")
            or _guess_zip_filename_from_url(url)
            or build_igme_brgm_zip_name(dataset_name, dataset_cfg)
        )

        zip_path = raw_dir / str(zip_filename)

        if download_cfg.get("enabled", True):
            _download_file(
                url=url,
                output_path=zip_path,
                overwrite=overwrite,
            )
        else:
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Download is disabled and ZIP does not exist: {zip_path}"
                )
            progress_log(f"[download] Download disabled, using existing ZIP: {zip_path}")
            progress_advance_stage_task(name=zip_path.name)

        output_paths.append(zip_path)

        if extract_after_download:
            extract_dir = raw_dir / "extracted" / safe_name(dataset_name)

            _extract_zip(
                zip_path=zip_path,
                extract_dir=extract_dir,
                overwrite=overwrite,
            )

            output_paths.append(extract_dir)

    return output_paths
