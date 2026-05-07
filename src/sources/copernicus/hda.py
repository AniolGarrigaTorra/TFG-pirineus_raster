from __future__ import annotations

from pathlib import Path
import inspect
import json
import re
import shutil
import time
from typing import Any


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

        print("[wekeo_hda] Creating client with explicit YAML credentials.")
        print("[wekeo_hda] WARNING: this is not recommended for committed configs.")
        conf = Configuration(user=username, password=password)
        return Client(config=conf, max_workers=max_workers)

    if config_path:
        print(f"[wekeo_hda] Creating client with config file: {config_path}")
        conf = Configuration(path=config_path)
        return Client(config=conf, max_workers=max_workers)

    print("[wekeo_hda] Creating client from ~/.hdarc or HDA_USER/HDA_PASSWORD.")
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
    print(f"[wekeo_hda] Directory inspection: {directory}")

    if not directory.exists():
        print("[wekeo_hda] Directory does not exist.")
        return

    entries = sorted(directory.rglob("*"))
    print(f"[wekeo_hda] Total entries under directory: {len(entries)}")

    for path in entries[:max_entries]:
        kind = "DIR " if path.is_dir() else "FILE"
        size = path.stat().st_size if path.is_file() else 0
        print(f"  [{kind}] {path} ({size} bytes)")

    if len(entries) > max_entries:
        print(f"  ... truncated, {len(entries) - max_entries} more entries")


def download_matches(matches: Any, tmp_dir: Path, overwrite: bool) -> None:
    tmp_dir = _normalise_tmp_dir(tmp_dir)
    _ensure_dir(tmp_dir)

    print(f"[wekeo_hda] Download dir absolute: {tmp_dir}")

    try:
        signature = inspect.signature(matches.download)
        print(f"[wekeo_hda] matches.download signature: {signature}")
    except Exception:
        signature = None

    if signature is not None and "force" in signature.parameters:
        matches.download(download_dir=str(tmp_dir), force=overwrite)
        return

    try:
        matches.download(download_dir=str(tmp_dir))
        return
    except TypeError:
        pass

    try:
        matches.download(str(tmp_dir))
        return
    except TypeError:
        pass

    raise RuntimeError(
        "Could not call matches.download with the known signatures. "
        "Please check your installed hda version with:\n"
        "  python -c \"import hda; print(hda.__version__)\""
    )


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
        print(f"[wekeo_hda] Result IDs written: {result_ids_path}")
    except Exception as exc:
        print(f"[wekeo_hda] Could not write result IDs: {exc}")


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
    """
    download_cfg = source_cfg.get("download", {}) or {}
    hda_cfg = download_cfg.get("hda", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    output_path = Path(output_path)

    query = load_hda_query(spec)

    variable = str(spec.get("variable", "unknown_variable"))

    configured_tmp_root = hda_cfg.get(
        "temporary_dir",
        output_path.parent / "_hda_downloads",
    )
    tmp_root = _normalise_tmp_dir(configured_tmp_root)
    tmp_dir = tmp_root / variable

    if tmp_dir.exists() and overwrite:
        shutil.rmtree(tmp_dir)

    _ensure_dir(tmp_dir)

    print("==============================")
    print(f"[wekeo_hda] Variable: {variable}")
    print(f"[wekeo_hda] Output path: {output_path}")
    print(f"[wekeo_hda] Temporary root: {tmp_root}")
    print(f"[wekeo_hda] Temporary dir: {tmp_dir}")
    print(f"[wekeo_hda] Dataset ID: {query.get('dataset_id')}")
    print("==============================")

    before_files = set(find_downloaded_files(tmp_dir))

    client = create_hda_client(download_cfg)

    max_results = spec.get("max_results", hda_cfg.get("max_results"))
    if max_results is not None:
        max_results = int(max_results)
        print(f"[wekeo_hda] Searching with max_results={max_results}")
        matches = client.search(query, max_results)
    else:
        print("[wekeo_hda] Searching without max_results limit")
        matches = client.search(query)

    print("[wekeo_hda] Search results:")
    print(matches)

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

    print("[wekeo_hda] Downloading...")
    download_matches(
        matches=matches,
        tmp_dir=tmp_dir,
        overwrite=overwrite,
    )

    sleep_after_download = int(hda_cfg.get("sleep_after_download_seconds", 2))
    if sleep_after_download > 0:
        time.sleep(sleep_after_download)

    file_pattern = spec.get("file_pattern")
    downloaded_files = find_downloaded_files(tmp_dir, pattern=file_pattern)

    if not downloaded_files and file_pattern:
        print(
            f"[wekeo_hda] No files matched file_pattern={file_pattern!r}. "
            "Retrying without pattern to inspect actual downloaded files."
        )
        downloaded_files = find_downloaded_files(tmp_dir, pattern=None)

    after_files = set(find_downloaded_files(tmp_dir))
    new_files = sorted(after_files - before_files)

    print(f"[wekeo_hda] Downloaded files found: {len(downloaded_files)}")
    for path in downloaded_files[:50]:
        print(f"  - {path}")

    print(f"[wekeo_hda] New files since start: {len(new_files)}")
    for path in new_files[:50]:
        print(f"  + {path}")

    if not downloaded_files:
        print_directory_tree(tmp_root)
        print_directory_tree(tmp_dir)
        raise FileNotFoundError(
            "HDA search returned results, but no downloaded files were found "
            "under the expected temporary directory.\n\n"
            f"Expected directory:\n  {tmp_dir}"
        )

    return downloaded_files