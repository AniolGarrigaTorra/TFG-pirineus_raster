from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.io.paths import ensure_dir
from src.sources.pdca.naming import raw_zip_path

USER_AGENT = "pirineus-raster-pipeline/0.1 (+PDCA Zenodo downloader)"


def _urlopen_json(url: str, timeout: int) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_zenodo_record(source_cfg: dict) -> dict:
    download_cfg = source_cfg.get("download", {})
    record_id = str(download_cfg.get("record_id", "1186639"))
    api_url = download_cfg.get(
        "api_url",
        f"https://zenodo.org/api/records/{record_id}",
    )
    timeout = int(download_cfg.get("timeout_seconds", 600))
    print(f"[pdca:download] Zenodo API: {api_url}")
    return _urlopen_json(api_url, timeout=timeout)


def _matches_file_policy(file_obj: dict, source_cfg: dict) -> bool:
    key = str(file_obj.get("key") or file_obj.get("filename") or "")
    key_lower = key.lower()
    download_cfg = source_cfg.get("download", {})

    include_extensions = [
        ext.lower()
        for ext in download_cfg.get("include_extensions", [".zip"])
    ]
    exclude_contains = [
        str(token).lower()
        for token in download_cfg.get("exclude_contains", [])
    ]
    include_contains = [
        str(token).lower()
        for token in download_cfg.get("include_contains", [])
    ]

    if include_extensions and not any(key_lower.endswith(ext) for ext in include_extensions):
        return False

    if any(token in key_lower for token in exclude_contains):
        return False

    if include_contains and not any(token in key_lower for token in include_contains):
        return False

    return True


def _download_url(file_obj: dict) -> str:
    links = file_obj.get("links", {})
    for candidate in ["self", "download", "content"]:
        if links.get(candidate):
            return links[candidate]
    if file_obj.get("download_url"):
        return file_obj["download_url"]
    raise KeyError(f"Could not find download URL in Zenodo file object: {file_obj}")


def _download_file(
    url: str,
    output_path: Path,
    overwrite: bool,
    timeout: int,
    max_retries: int,
    retry_sleep_seconds: int,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[pdca:download] Exists, skipping: {output_path}")
        return

    ensure_dir(output_path.parent)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    if temporary_path.exists():
        temporary_path.unlink()

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        print(f"[pdca:download] Attempt {attempt}/{max_retries}: {output_path.name}")
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                with temporary_path.open("wb") as f:
                    shutil.copyfileobj(response, f)
            temporary_path.rename(output_path)
            print(f"[pdca:download] Finished: {output_path}")
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if temporary_path.exists():
                temporary_path.unlink()
            print(f"[pdca:download] Failed: {exc}")
            if attempt < max_retries:
                print(f"[pdca:download] Sleeping {retry_sleep_seconds}s before retry...")
                time.sleep(retry_sleep_seconds)

    raise RuntimeError(
        "PDCA download failed after multiple attempts.\n"
        f"URL: {url}\nOutput: {output_path}\nLast error: {last_error}"
    )


def download_pdca_raw_files(source_cfg: dict, raw_dir: Path) -> list[Path]:
    ensure_dir(raw_dir)
    download_cfg = source_cfg.get("download", {})
    enabled = bool(download_cfg.get("enabled", True))
    overwrite = bool(download_cfg.get("overwrite_existing", False))
    timeout = int(download_cfg.get("timeout_seconds", 1800))
    max_retries = int(download_cfg.get("max_retries", 5))
    retry_sleep_seconds = int(download_cfg.get("retry_sleep_seconds", 60))

    if not enabled:
        existing = sorted(raw_dir.glob("*.zip"))
        if existing:
            print(f"[pdca:download] Automatic download disabled. Found {len(existing)} ZIPs.")
            return existing
        raise FileNotFoundError(
            "Automatic PDCA download is disabled and no ZIP files exist in "
            f"{raw_dir}"
        )

    record = fetch_zenodo_record(source_cfg)
    files = record.get("files", [])
    selected = [file_obj for file_obj in files if _matches_file_policy(file_obj, source_cfg)]

    if not selected:
        available = "\n".join(str(f.get("key") or f.get("filename")) for f in files)
        raise FileNotFoundError(
            "No Zenodo files matched the PDCA download policy.\n"
            f"Available files:\n{available}"
        )

    print(f"[pdca:download] Selected files: {len(selected)}")
    paths: list[Path] = []

    for file_obj in selected:
        key = str(file_obj.get("key") or file_obj.get("filename"))
        output_path = raw_zip_path(raw_dir, key)
        url = _download_url(file_obj)
        size = file_obj.get("size")
        print(f"[pdca:download] File: {key} size={size}")
        _download_file(
            url=url,
            output_path=output_path,
            overwrite=overwrite,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        paths.append(output_path)

    return paths
