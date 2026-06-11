from __future__ import annotations

from pathlib import Path
import inspect
import json
import re
import shutil
import time
from typing import Any

from src.pipeline.progress import progress_log


def _format_bytes(value: float | int | None) -> str:
    """Format bytes to human readable format."""
    if value is None:
        return "?"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_duration(seconds: float | None) -> str:
    """Format seconds to HH:MM:SS format."""
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _import_hda():
    try:
        from hda import Client, Configuration  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The WEkEO HDA client is not installed.\n\n"
            "Install it with one of:\n"
            "  mamba install conda-forge::hda\n"
            "  pip install hda -U\n\n"
            "Then configure your WEkEO credentials with ~/.hdarc or "
            "HDA_USER/HDA_PASSWORD environment variables."
        ) from exc

    return Client, Configuration


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalise_tmp_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_hda_query(spec: dict[str, Any]) -> dict[str, Any]:
    query = spec.get("hda_query")
    query_path = spec.get("hda_query_path")

    if query is not None and query_path is not None:
        raise ValueError("Use either hda_query or hda_query_path, not both.")

    if query is not None:
        if not isinstance(query, dict):
            raise TypeError("hda_query must be a dictionary.")
        return query

    if query_path is not None:
        path = Path(query_path)
        if not path.exists():
            raise FileNotFoundError(f"HDA query file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded, dict):
            raise TypeError(f"HDA query file must contain a JSON object: {path}")

        return loaded

    raise ValueError(
        "Missing HDA query. Add either hda_query or hda_query_path "
        "under download.files.<variable>."
    )


def create_hda_client(download_cfg: dict[str, Any]):
    Client, Configuration = _import_hda()

    hda_cfg = download_cfg.get("hda", {}) or {}

    username = hda_cfg.get("username")
    password = hda_cfg.get("password")
    config_path = hda_cfg.get("config_path")
    max_workers = int(hda_cfg.get("max_workers", 2))

    if username or password:
        if not username or not password:
            raise ValueError(
                "Both download.hda.username and download.hda.password "
                "must be set if using explicit credentials."
            )

        progress_log("[wekeo_hda] Creating client with explicit YAML credentials.")
        progress_log("[wekeo_hda] WARNING: this is not recommended for committed configs.", level="warning")
        conf = Configuration(user=username, password=password)
        return Client(config=conf, max_workers=max_workers)

    if config_path:
        progress_log(f"[wekeo_hda] Creating client with config file: {config_path}")
        conf = Configuration(path=config_path)
        return Client(config=conf, max_workers=max_workers)

    progress_log("[wekeo_hda] Creating client from ~/.hdarc or HDA_USER/HDA_PASSWORD.")
    return Client(max_workers=max_workers)


def find_downloaded_files(
    directory: Path,
    pattern: str | None = None,
) -> list[Path]:
    directory = Path(directory)

    if not directory.exists():
        return []

    all_files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.endswith(".part")
        and not path.name.endswith(".tmp")
        and not path.name.endswith(".crdownload")
    ]

    if pattern:
        filtered: list[Path] = []
        for path in all_files:
            relative = path.relative_to(directory)
            if relative.match(pattern) or path.match(pattern) or path.name == pattern:
                filtered.append(path)
        return sorted(filtered)

    return sorted(all_files)


def print_directory_tree(directory: Path, max_entries: int = 100) -> None:
    progress_log(f"[wekeo_hda] Directory inspection: {directory}")

    if not directory.exists():
        progress_log("[wekeo_hda] Directory does not exist.")
        return

    entries = sorted(directory.rglob("*"))
    progress_log(f"[wekeo_hda] Total entries under directory: {len(entries)}")

    for path in entries[:max_entries]:
        kind = "DIR " if path.is_dir() else "FILE"
        size = path.stat().st_size if path.is_file() else 0
        progress_log(f"  [{kind}] {path} ({size} bytes)")

    if len(entries) > max_entries:
        progress_log(f"  ... truncated, {len(entries) - max_entries} more entries")


def download_matches(
    matches: Any,
    tmp_dir: Path,
    overwrite: bool,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> None:
    """
    Download files with smart retry logic and exponential backoff.
    
    Parameters:
    -----------
    matches : Any
        HDA search results object
    tmp_dir : Path
        Temporary directory for downloads
    overwrite : bool
        Whether to force overwrite existing files
    max_retries : int
        Maximum number of retry attempts (default: 3)
    retry_backoff : float
        Backoff multiplier between retries (default: 2.0 for exponential)
    """
    tmp_dir = _normalise_tmp_dir(tmp_dir)
    _ensure_dir(tmp_dir)

    progress_log(f"[wekeo_hda] Download dir absolute: {tmp_dir}")

    try:
        signature = inspect.signature(matches.download)
        progress_log(f"[wekeo_hda] matches.download signature: {signature}")
    except Exception:
        signature = None

    # Track files downloaded to detect silent failures
    files_before = set(find_downloaded_files(tmp_dir))

    # Determine download method
    download_method = None
    if signature is not None and "force" in signature.parameters:
        download_method = lambda: matches.download(download_dir=str(tmp_dir), force=overwrite)
    elif signature is not None and "download_dir" in signature.parameters:
        download_method = lambda: matches.download(download_dir=str(tmp_dir))
    else:
        download_method = lambda: matches.download(str(tmp_dir))

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                retry_delay = int(2 ** (attempt - 2)) * int(retry_backoff)
                progress_log(
                    f"[wekeo_hda] Retry attempt {attempt}/{max_retries} "
                    f"(waiting {retry_delay}s before retry)...",
                    level="warning"
                )
                time.sleep(retry_delay)
            else:
                progress_log(f"[wekeo_hda] Attempting download (attempt 1/{max_retries})...")

            download_method()

            # Verify files were actually downloaded (detect silent failures)
            files_after = set(find_downloaded_files(tmp_dir))
            new_files = files_after - files_before

            if new_files or attempt == max_retries:
                # Success: files downloaded OR last attempt (accept result)
                if new_files:
                    progress_log(f"[wekeo_hda] ✓ Download successful ({len(new_files)} new files)")
                else:
                    progress_log(
                        f"[wekeo_hda] ⚠ Download completed but no new files detected "
                        f"(may indicate silent failure - but accepting on final attempt)",
                        level="warning"
                    )
                return

            # No files downloaded - retry if not last attempt
            if attempt < max_retries:
                progress_log(
                    f"[wekeo_hda] ⚠ Download returned no files, will retry "
                    f"({len(files_before)} files present)",
                    level="warning"
                )
                continue

        except TypeError as e:
            # Known signature issue - try next method
            last_error = e
            if attempt < max_retries:
                progress_log(
                    f"[wekeo_hda] Download method signature mismatch, retrying...",
                    level="warning"
                )
                continue
        except Exception as e:
            last_error = e
            progress_log(
                f"[wekeo_hda] Download attempt {attempt} failed: {e}",
                level="warning"
            )
            if attempt < max_retries:
                continue

    # All retries exhausted
    error_msg = (
        "WEkEO HDA download failed after all retry attempts.\n\n"
        "Retry summary:\n"
        f"  Max retries: {max_retries}\n"
        f"  Backoff multiplier: {retry_backoff}\n\n"
        "Troubleshooting:\n"
        "  1. Check HDA version: python -c \"import hda; print(hda.__version__)\"\n"
        "  2. Check WEkEO service status\n"
        "  3. Reduce max_workers in config\n"
        "  4. Check authentication: hda --show-token\n"
        "  5. Try manual download from https://dataspace.copernicus.eu/"
    )
    if last_error:
        error_msg += f"\n\nLast error: {last_error}"

    raise RuntimeError(error_msg)


def get_search_result_count(matches: Any) -> int | None:
    results = getattr(matches, "results", None)

    if results is not None:
        try:
            return len(results)
        except TypeError:
            pass

    for attr in ("items", "total", "count", "numberMatched"):
        value = getattr(matches, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass

    text = str(matches)
    match = re.search(r"items=(\d+)", text)
    if match:
        return int(match.group(1))

    return None


def write_hda_manifests(
    *,
    output_path: Path,
    query: dict[str, Any],
    matches: Any,
) -> None:
    query_manifest_path = output_path.with_suffix(output_path.suffix + ".hda_query.json")
    query_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with query_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(query, f, indent=2, ensure_ascii=False)

    result_ids_path = output_path.with_suffix(output_path.suffix + ".hda_results.txt")

    try:
        with result_ids_path.open("w", encoding="utf-8") as f:
            for item in matches.results:
                f.write(str(item.get("id", item)) + "\n")
        progress_log(f"[wekeo_hda] Result IDs written: {result_ids_path}")
    except Exception as exc:
        progress_log(f"[wekeo_hda] Could not write result IDs: {exc}", level="warning")


def download_with_wekeo_hda(
    *,
    source_cfg: dict[str, Any],
    spec: dict[str, Any],
    output_path: Path,
) -> list[Path]:
    """
    Search and download files through WEkEO HDA.

    This function intentionally does not postprocess downloaded files.
    It only returns the downloaded file paths.
    
    Features:
    - Smart caching: checks if final output already exists
    - Reuses temporary downloads if available
    - Detailed logging for diagnostics
    """
    download_cfg = source_cfg.get("download", {}) or {}
    hda_cfg = download_cfg.get("hda", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    output_path = Path(output_path)
    
    variable = str(spec.get("variable", "unknown_variable"))

    # =========================================================================
    # NOTE: Output file cache should be checked at download.py level
    # before calling this function. This check is only a safety net.
    # If output exists, return empty list (no raw files to process)
    # =========================================================================
    if output_path.exists() and not overwrite:
        progress_log(f"[wekeo_hda] Cache hit: {output_path}")
        return []

    configured_tmp_root = hda_cfg.get(
        "temporary_dir",
        output_path.parent / "_hda_downloads",
    )
    tmp_root = _normalise_tmp_dir(configured_tmp_root)
    tmp_dir = tmp_root / variable

    if tmp_dir.exists() and overwrite:
        progress_log(f"[wekeo_hda] Removing existing temp directory (overwrite=True): {tmp_dir}")
        shutil.rmtree(tmp_dir)

    _ensure_dir(tmp_dir)

    # =========================================================================
    # CACHING: Check if temporary files already exist from previous downloads
    # =========================================================================
    existing_tmp_files = find_downloaded_files(tmp_dir, pattern=None)
    if existing_tmp_files and not overwrite:
        progress_log(
            f"[wekeo_hda] ✓ TEMP CACHE: Found {len(existing_tmp_files)} files from previous download"
        )
        for path in existing_tmp_files[:10]:
            progress_log(f"[wekeo_hda]   - {path.name} ({_format_bytes(path.stat().st_size)})")
        if len(existing_tmp_files) > 10:
            progress_log(f"[wekeo_hda]   ... and {len(existing_tmp_files) - 10} more")
        return existing_tmp_files

    query = load_hda_query(spec)

    progress_log(f"[wekeo_hda] Variable: {variable}")
    progress_log(f"[wekeo_hda] Output path: {output_path}")
    progress_log(f"[wekeo_hda] Temporary root: {tmp_root}")
    progress_log(f"[wekeo_hda] Temporary dir: {tmp_dir}")
    progress_log(f"[wekeo_hda] Dataset ID: {query.get('dataset_id')}")

    before_files = set(find_downloaded_files(tmp_dir))

    # =========================================================================
    # HDA SEARCH & DOWNLOAD
    # =========================================================================
    client = create_hda_client(download_cfg)

    max_results = spec.get("max_results", hda_cfg.get("max_results"))
    if max_results is not None:
        max_results = int(max_results)
        progress_log(f"[wekeo_hda] Searching with max_results={max_results}...")
        matches = client.search(query, max_results)
    else:
        progress_log("[wekeo_hda] Searching without max_results limit...")
        matches = client.search(query)

    progress_log(f"[wekeo_hda] Search results: {matches}")

    result_count = get_search_result_count(matches)
    if result_count == 0:
        query_manifest_path = output_path.with_suffix(
            output_path.suffix + ".hda_query.json"
        )
        query_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with query_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(query, f, indent=2, ensure_ascii=False)

        raise RuntimeError(
            "WEkEO HDA search returned 0 results.\n\n"
            f"Variable: {variable}\n"
            f"Dataset ID: {query.get('dataset_id')}\n"
            f"Query manifest: {query_manifest_path}\n\n"
            "This is not a download or directory problem. "
            "The hda_query does not match any downloadable product."
        )

    write_hda_manifests(
        output_path=output_path,
        query=query,
        matches=matches,
    )

    progress_log(f"[wekeo_hda] Downloading {result_count} items through HDA client...")
    
    # Get retry configuration
    max_retries = int(hda_cfg.get("max_retries", 3))
    retry_backoff = float(hda_cfg.get("retry_backoff_multiplier", 2.0))
    
    download_matches(
        matches=matches,
        tmp_dir=tmp_dir,
        overwrite=overwrite,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )

    sleep_after_download = int(hda_cfg.get("sleep_after_download_seconds", 2))
    if sleep_after_download > 0:
        progress_log(f"[wekeo_hda] Sleeping {sleep_after_download}s after download...")
        time.sleep(sleep_after_download)

    file_pattern = spec.get("file_pattern")
    downloaded_files = find_downloaded_files(tmp_dir, pattern=file_pattern)

    if not downloaded_files and file_pattern:
        progress_log(
            f"[wekeo_hda] No files matched file_pattern={file_pattern!r}. "
            "Retrying without pattern to inspect actual downloaded files."
        )
        downloaded_files = find_downloaded_files(tmp_dir, pattern=None)

    after_files = set(find_downloaded_files(tmp_dir))
    new_files = sorted(after_files - before_files)

    progress_log(f"[wekeo_hda] ✓ Downloaded files found: {len(downloaded_files)}")
    total_size = 0
    for path in downloaded_files[:20]:
        size = path.stat().st_size
        total_size += size
        progress_log(f"[wekeo_hda]   - {path.name} ({_format_bytes(size)})")
    if len(downloaded_files) > 20:
        progress_log(f"[wekeo_hda]   ... and {len(downloaded_files) - 20} more")
    progress_log(f"[wekeo_hda] Total size: {_format_bytes(total_size)}")

    progress_log(f"[wekeo_hda] New files since start: {len(new_files)}")
    for path in new_files[:10]:
        progress_log(f"[wekeo_hda]   + {path.name}")

    if not downloaded_files:
        print_directory_tree(tmp_root)
        print_directory_tree(tmp_dir)
        
        # Enhanced diagnostics
        search_result_count = get_search_result_count(matches) or 0
        diagnostic_msg = (
            "HDA search returned results, but no downloaded files were found "
            "under the expected temporary directory.\n\n"
            f"Search reported {search_result_count} items available.\n"
            f"Expected directory: {tmp_dir}\n\n"
            "Possible causes:\n"
            "  1. WEkEO download service timeout or failure\n"
            "  2. Network interruption during transfer of large batch ({}, {}).\n"
            "  3. HDA API data inconsistency (search returns stale results)\n"
            "  4. File format/naming mismatch\n\n"
            "Recommended actions:\n"
            "  - Check WEkEO service status\n"
            "  - Try reducing max_workers in config (current: {}) \n"
            "  - Check HDA version: python -c \"import hda; print(hda.__version__)\"\n"
            "  - Check authentication: hda --show-token\n"
            "  - Try manual download from https://dataspace.copernicus.eu/"
        )
        
        # Get file pattern for diagnostic message
        file_pattern_str = spec.get("file_pattern", "unknown")
        volume_str = getattr(matches, "volume", "unknown") if hasattr(matches, 'volume') else "unknown"
        max_workers = hda_cfg.get("max_workers", 2)
        
        raise FileNotFoundError(
            diagnostic_msg.format(
                file_pattern_str,
                volume_str,
                max_workers,
            )
        )

    return downloaded_files
