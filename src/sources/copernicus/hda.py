from __future__ import annotations

from pathlib import Path
import json
import shutil
import time
from typing import Any


def _import_hda():
    """
    Import hda lazily so the rest of the project can still work
    even if the optional WEkEO dependency is not installed.
    """
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


def _load_hda_query(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Load HDA query from either:
      - spec["hda_query"] as an inline YAML dict
      - spec["hda_query_path"] as a JSON file path

    Inline YAML is preferred for reproducibility inside source configs.
    """
    query = spec.get("hda_query")
    query_path = spec.get("hda_query_path")

    if query is not None and query_path is not None:
        raise ValueError(
            "Use either hda_query or hda_query_path, not both."
        )

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


def _create_hda_client(download_cfg: dict[str, Any]):
    """
    Create HDA client.

    Supported authentication patterns:
      1. default ~/.hdarc
      2. environment variables HDA_USER/HDA_PASSWORD
      3. explicit hda_config_path in YAML
      4. explicit username/password in YAML, discouraged
    """
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


def _find_downloaded_files(
    directory: Path,
    pattern: str | None = None,
) -> list[Path]:
    """
    Find files downloaded by HDA.

    If pattern is provided, it is treated as a simple shell glob over filenames,
    not a regex.
    """
    all_files = [
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.name.endswith(".part")
        and not path.name.endswith(".tmp")
    ]

    if pattern:
        return sorted(
            path
            for path in all_files
            if path.match(pattern) or path.name == pattern
        )

    return sorted(all_files)


def _copy_or_move_single_result(
    downloaded_files: list[Path],
    output_path: Path,
    move: bool,
    overwrite: bool,
) -> Path:
    """
    Copy or move the single downloaded result to the raw file expected by the
    rest of the pipeline.
    """
    if output_path.exists() and not overwrite:
        print(f"[wekeo_hda] Exists, skipping final copy: {output_path}")
        return output_path

    if len(downloaded_files) == 0:
        raise FileNotFoundError(
            "HDA download completed but no files were found in the temporary "
            "download directory."
        )

    if len(downloaded_files) > 1:
        listing = "\n".join(str(p) for p in downloaded_files[:50])
        raise RuntimeError(
            "HDA query downloaded more than one file, but this source variable "
            "expects exactly one raw file.\n\n"
            "Fix one of these:\n"
            "  - make the hda_query more restrictive\n"
            "  - set file_pattern to select one file\n"
            "  - set allow_multiple: true and adapt later stages\n\n"
            f"Downloaded files:\n{listing}"
        )

    source_path = downloaded_files[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and overwrite:
        output_path.unlink()

    print(f"[wekeo_hda] Source downloaded file: {source_path}")
    print(f"[wekeo_hda] Final raw file: {output_path}")

    if move:
        shutil.move(str(source_path), str(output_path))
    else:
        shutil.copy2(source_path, output_path)

    return output_path


def download_with_wekeo_hda(
    *,
    source_cfg: dict[str, Any],
    spec: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Download one Copernicus file using the WEkEO HDA Python client.

    YAML example:

    download:
      enabled: true
      mode: wekeo_hda
      hda:
        max_workers: 2
      files:
        tree_cover_density:
          filename: tree_cover_density_2021_10m.tif
          hda_query:
            dataset_id: "..."
            bbox: [...]
            ...
          max_results: 1
          file_pattern: "*.tif"
    """
    download_cfg = source_cfg.get("download", {}) or {}
    hda_cfg = download_cfg.get("hda", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))

    if output_path.exists() and not overwrite:
        print(f"[wekeo_hda] Exists, skipping: {output_path}")
        return output_path

    query = _load_hda_query(spec)

    variable = spec.get("variable", "unknown_variable")
    tmp_root = Path(
        hda_cfg.get(
            "temporary_dir",
            output_path.parent / "_hda_downloads",
        )
    )
    tmp_dir = tmp_root / variable

    if tmp_dir.exists() and overwrite:
        shutil.rmtree(tmp_dir)

    _ensure_dir(tmp_dir)

    query_manifest_path = output_path.with_suffix(output_path.suffix + ".hda_query.json")
    query_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with query_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(query, f, indent=2, ensure_ascii=False)

    print("==============================")
    print(f"[wekeo_hda] Variable: {variable}")
    print(f"[wekeo_hda] Output path: {output_path}")
    print(f"[wekeo_hda] Temporary dir: {tmp_dir}")
    print(f"[wekeo_hda] Query manifest: {query_manifest_path}")
    print(f"[wekeo_hda] Dataset ID: {query.get('dataset_id')}")
    print("==============================")

    client = _create_hda_client(download_cfg)

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

    result_ids_path = output_path.with_suffix(output_path.suffix + ".hda_results.txt")
    try:
        with result_ids_path.open("w", encoding="utf-8") as f:
            for item in matches.results:
                f.write(str(item.get("id", item)) + "\n")
        print(f"[wekeo_hda] Result IDs written: {result_ids_path}")
    except Exception as exc:
        print(f"[wekeo_hda] Could not write result IDs: {exc}")

    print("[wekeo_hda] Downloading...")
    matches.download(download_dir=str(tmp_dir))

    sleep_after_download = int(hda_cfg.get("sleep_after_download_seconds", 0))
    if sleep_after_download > 0:
        time.sleep(sleep_after_download)

    file_pattern = spec.get("file_pattern")
    downloaded_files = _find_downloaded_files(tmp_dir, pattern=file_pattern)

    print(f"[wekeo_hda] Downloaded files found: {len(downloaded_files)}")
    for path in downloaded_files[:20]:
        print(f"  - {path}")

    allow_multiple = bool(spec.get("allow_multiple", False))
    move_downloaded = bool(hda_cfg.get("move_downloaded_to_raw", False))

    if allow_multiple:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        marker = output_path.with_suffix(output_path.suffix + ".multiple_downloads.txt")
        with marker.open("w", encoding="utf-8") as f:
            for path in downloaded_files:
                f.write(str(path) + "\n")

        print(
            "[wekeo_hda] allow_multiple=true, leaving downloaded files in "
            f"{tmp_dir} and writing marker {marker}"
        )
        return marker

    return _copy_or_move_single_result(
        downloaded_files=downloaded_files,
        output_path=output_path,
        move=move_downloaded,
        overwrite=overwrite,
    )