from __future__ import annotations

from pathlib import Path
import inspect
import json
import re
import shutil
import time
import zipfile
from typing import Any

import rasterio
from rasterio.merge import merge


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


def _normalise_tmp_dir(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_hda_query(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Load HDA query from either:
      - spec["hda_query"] as an inline YAML dict
      - spec["hda_query_path"] as a JSON file path
    """
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


def _create_hda_client(download_cfg: dict[str, Any]):
    """
    Create HDA client.

    Supported authentication patterns:
      1. default ~/.hdarc
      2. environment variables HDA_USER/HDA_PASSWORD
      3. explicit config_path in YAML
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

    If pattern is provided, it is treated as a shell glob. It is matched against:
      - full path relative to directory
      - filename only
    """
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


def _print_directory_tree(directory: Path, max_entries: int = 100) -> None:
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


def _download_matches(matches: Any, tmp_dir: Path, overwrite: bool) -> None:
    """
    Download HDA matches using a version-tolerant call.

    Different hda versions expose slightly different signatures.
    """
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


def _find_tif_members_in_zip(
    zip_path: Path,
    zip_member_pattern: str | None = None,
) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()

    tif_members = [
        member
        for member in members
        if member.lower().endswith((".tif", ".tiff"))
    ]

    if zip_member_pattern:
        pattern = re.compile(zip_member_pattern)
        tif_members = [
            member
            for member in tif_members
            if pattern.search(member) or pattern.search(Path(member).name)
        ]

    return sorted(tif_members)


def _zip_member_to_rasterio_uri(zip_path: Path, member: str) -> str:
    return f"zip://{zip_path}!{member}"


def _mosaic_zip_geotiffs(
    *,
    zip_paths: list[Path],
    output_path: Path,
    zip_member_pattern: str | None,
    overwrite: bool,
    compression: str = "LZW",
    allow_multiple_zip_members: bool = False,
    skip_zip_without_matching_members: bool = False,
) -> Path:
    """
    Build a single mosaic GeoTIFF from one or more ZIP files containing GeoTIFFs.

    Supports two cases:
      1. one TIFF per ZIP, the default for most CLMS tiled products
      2. multiple TIFFs per ZIP, needed for products such as Water & Wetness
         or CORINE packages
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[wekeo_hda] Mosaic exists, skipping: {output_path}")
        return output_path

    if not zip_paths:
        raise FileNotFoundError("No ZIP files provided for mosaic_zip_geotiff.")

    raster_uris: list[str] = []

    for zip_path in zip_paths:
        members = _find_tif_members_in_zip(
            zip_path=zip_path,
            zip_member_pattern=zip_member_pattern,
        )

        if len(members) == 0:
            message = (
                f"No GeoTIFF members found in {zip_path} "
                f"with zip_member_pattern={zip_member_pattern!r}"
            )

            if skip_zip_without_matching_members:
                print(f"[wekeo_hda] {message}. Skipping ZIP.")
                continue

            raise FileNotFoundError(message)

        if len(members) > 1 and not allow_multiple_zip_members:
            listing = "\n".join(members[:30])
            raise RuntimeError(
                f"More than one GeoTIFF member found in {zip_path}.\n"
                "Please set a more restrictive zip_member_pattern, or set:\n"
                "  allow_multiple_zip_members: true\n"
                "if all matching TIFFs should be mosaicked.\n"
                f"Members:\n{listing}"
            )

        for member in members:
            raster_uris.append(_zip_member_to_rasterio_uri(zip_path, member))

    if not raster_uris:
        raise FileNotFoundError(
            "No GeoTIFF members were selected for the mosaic after applying "
            f"zip_member_pattern={zip_member_pattern!r}. "
            "Check the pattern or the downloaded ZIP contents."
        )

    print("[wekeo_hda] Building GeoTIFF mosaic")
    print(f"[wekeo_hda] ZIP files: {len(zip_paths)}")
    print(f"[wekeo_hda] Raster members selected: {len(raster_uris)}")
    print(f"[wekeo_hda] Output mosaic: {output_path}")

    srcs = []
    try:
        for uri in raster_uris:
            print(f"  - {uri}")
            srcs.append(rasterio.open(uri))

        mosaic, transform = merge(srcs)

        profile = srcs[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            compress=compression,
            BIGTIFF="IF_SAFER",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and overwrite:
            output_path.unlink()

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
            dst.update_tags(
                source="WEkEO HDA",
                postprocess="mosaic_zip_geotiff",
                source_zip_count=str(len(zip_paths)),
                selected_raster_count=str(len(raster_uris)),
            )

    finally:
        for src in srcs:
            src.close()

    print(f"[wekeo_hda] Mosaic written: {output_path}")
    return output_path


def _copy_or_move_single_result(
    downloaded_files: list[Path],
    output_path: Path,
    move: bool,
    overwrite: bool,
) -> Path:
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
            "Use one of these options:\n"
            "  - make the hda_query more restrictive\n"
            "  - set file_pattern to select one file\n"
            "  - set postprocess: mosaic_zip_geotiff for tiled ZIP GeoTIFF products\n\n"
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

def _get_search_result_count(matches: Any) -> int | None:
    """
    Try to infer the number of HDA search results in a version-tolerant way.
    """
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


def download_with_wekeo_hda(
    *,
    source_cfg: dict[str, Any],
    spec: dict[str, Any],
    output_path: Path,
) -> Path:
    """
    Download one Copernicus variable using the WEkEO HDA Python client.

    Supported postprocess modes:
      - none / missing: expects exactly one downloaded file
      - mosaic_zip_geotiff: expects one or more ZIP files containing GeoTIFFs,
        then builds a single raw GeoTIFF mosaic.
    """
    download_cfg = source_cfg.get("download", {}) or {}
    hda_cfg = download_cfg.get("hda", {}) or {}

    overwrite = bool(download_cfg.get("overwrite_existing", False))
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        print(f"[wekeo_hda] Exists, skipping: {output_path}")
        return output_path

    query = _load_hda_query(spec)

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

    query_manifest_path = output_path.with_suffix(output_path.suffix + ".hda_query.json")
    query_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with query_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(query, f, indent=2, ensure_ascii=False)

    print("==============================")
    print(f"[wekeo_hda] Variable: {variable}")
    print(f"[wekeo_hda] Output path: {output_path}")
    print(f"[wekeo_hda] Temporary root: {tmp_root}")
    print(f"[wekeo_hda] Temporary dir: {tmp_dir}")
    print(f"[wekeo_hda] Query manifest: {query_manifest_path}")
    print(f"[wekeo_hda] Dataset ID: {query.get('dataset_id')}")
    print(f"[wekeo_hda] Postprocess: {spec.get('postprocess')}")
    print("==============================")

    before_files = set(_find_downloaded_files(tmp_dir))

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

    result_count = _get_search_result_count(matches)

    if result_count == 0:
        raise RuntimeError(
            "WEkEO HDA search returned 0 results.\n\n"
            f"Variable: {variable}\n"
            f"Dataset ID: {query.get('dataset_id')}\n"
            f"Query manifest: {query_manifest_path}\n\n"
            "This is not a download or directory problem. "
            "The hda_query does not match any downloadable product. "
            "Copy the exact API request from the WEkEO Data Viewer and update "
            "download.files.<variable>.hda_query in the YAML."
        )

    result_ids_path = output_path.with_suffix(output_path.suffix + ".hda_results.txt")
    try:
        with result_ids_path.open("w", encoding="utf-8") as f:
            for item in matches.results:
                f.write(str(item.get("id", item)) + "\n")
        print(f"[wekeo_hda] Result IDs written: {result_ids_path}")
    except Exception as exc:
        print(f"[wekeo_hda] Could not write result IDs: {exc}")

    print("[wekeo_hda] Downloading...")
    _download_matches(
        matches=matches,
        tmp_dir=tmp_dir,
        overwrite=overwrite,
    )

    sleep_after_download = int(hda_cfg.get("sleep_after_download_seconds", 2))
    if sleep_after_download > 0:
        time.sleep(sleep_after_download)

    file_pattern = spec.get("file_pattern")
    downloaded_files = _find_downloaded_files(tmp_dir, pattern=file_pattern)

    if not downloaded_files and file_pattern:
        print(
            f"[wekeo_hda] No files matched file_pattern={file_pattern!r}. "
            "Retrying without pattern to inspect actual downloaded files."
        )
        downloaded_files = _find_downloaded_files(tmp_dir, pattern=None)

    after_files = set(_find_downloaded_files(tmp_dir))
    new_files = sorted(after_files - before_files)

    print(f"[wekeo_hda] Downloaded files found: {len(downloaded_files)}")
    for path in downloaded_files[:50]:
        print(f"  - {path}")

    print(f"[wekeo_hda] New files since start: {len(new_files)}")
    for path in new_files[:50]:
        print(f"  + {path}")

    if not downloaded_files:
        _print_directory_tree(tmp_root)
        _print_directory_tree(tmp_dir)

        raise FileNotFoundError(
            "HDA search returned results, but no downloaded files were found "
            "under the expected temporary directory.\n\n"
            f"Expected directory:\n  {tmp_dir}"
        )

    postprocess = spec.get("postprocess")

    if postprocess == "mosaic_zip_geotiff":
        zip_paths = [
            path
            for path in downloaded_files
            if path.suffix.lower() == ".zip"
        ]

        if not zip_paths:
            raise FileNotFoundError(
                "postprocess=mosaic_zip_geotiff was requested, "
                "but no ZIP files were downloaded."
            )

        return _mosaic_zip_geotiffs(
            zip_paths=zip_paths,
            output_path=output_path,
            zip_member_pattern=spec.get("zip_member_pattern"),
            overwrite=overwrite,
            compression=str(hda_cfg.get("mosaic_compression", "LZW")),
            allow_multiple_zip_members=bool(
                spec.get("allow_multiple_zip_members", False)
            ),
            skip_zip_without_matching_members=bool(
                spec.get("skip_zip_without_matching_members", False)
            ),
        )

    if postprocess not in (None, "", "none", "copy_single"):
        raise NotImplementedError(
            f"Unsupported WEkEO HDA postprocess={postprocess!r}. "
            "Supported: none, copy_single, mosaic_zip_geotiff"
        )

    move_downloaded = bool(hda_cfg.get("move_downloaded_to_raw", False))

    return _copy_or_move_single_result(
        downloaded_files=downloaded_files,
        output_path=output_path,
        move=move_downloaded,
        overwrite=overwrite,
    )