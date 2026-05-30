from __future__ import annotations

from pathlib import Path

from src.io.paths import ensure_dir
from src.sources.copernicus.download import (
    copy_local_file,
    download_file,
    download_multiple_files,
)
from src.sources.copernicus.postprocess import run_static_postprocess
from src.sources.generic_raster.naming import (
    get_download_file_specs,
    validate_generic_raster_source_config,
)


def _run_postprocess(
    *,
    input_paths: list[Path],
    output_path: Path,
    source_cfg: dict,
    spec: dict,
) -> list[Path]:
    written = run_static_postprocess(
        postprocess=spec.get("postprocess"),
        input_paths=input_paths,
        output_path=output_path,
        spec=spec,
        source_cfg=source_cfg,
    )
    return [written]


def download_generic_raster_raw_files(
    source_cfg: dict,
    raw_dir: Path,
    provider: str | None = None,
) -> list[Path]:
    validate_generic_raster_source_config(source_cfg, provider=provider)

    raw_dir = Path(raw_dir)
    ensure_dir(raw_dir)

    download_cfg = source_cfg.get("download", {}) or {}
    enabled = bool(download_cfg.get("enabled", True))
    mode = str(download_cfg.get("mode", "manual")).lower()
    overwrite = bool(download_cfg.get("overwrite_existing", False))

    print("[download] Generic raster raw dir:", raw_dir)
    print("[download] Provider:", source_cfg["source"]["provider"])
    print("[download] Mode:", mode)
    print("[download] Enabled:", enabled)

    raw_paths: list[Path] = []

    for spec in get_download_file_specs(source_cfg):
        variable = spec["variable"]
        output_path = raw_dir / spec["filename"]

        print("==============================")
        print(f"[download] Variable/download spec: {variable}")
        print(f"[download] File: {output_path}")

        if not enabled or mode == "manual":
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Expected raw file does not exist: {output_path}\n"
                    "Place the file there manually, switch to mode=local_file, "
                    "or configure download.files.<name>.url/urls with mode=manual_url."
                )
            print(f"[download] Manual file found: {output_path}")
            raw_paths.append(output_path)
            continue

        if mode == "manual_url":
            url = spec.get("url")
            urls = spec.get("urls")
            if urls:
                downloaded = download_multiple_files(
                    urls=urls,
                    output_dir=raw_dir / "_parts" / variable,
                    filenames=spec.get("filenames"),
                    overwrite=overwrite,
                )
                raw_paths.extend(
                    _run_postprocess(
                        input_paths=downloaded,
                        output_path=output_path,
                        source_cfg=source_cfg,
                        spec=spec,
                    )
                )
                continue
            if url:
                download_file(
                    url=url,
                    output_path=output_path,
                    overwrite=overwrite,
                )
                raw_paths.append(output_path)
                continue
            raise ValueError(
                f"Missing url or urls for variable={variable!r} in download.files"
            )

        if mode == "local_file":
            local_path = spec.get("local_path")
            if not local_path:
                raise ValueError(
                    f"Missing local_path for variable={variable!r} in download.files"
                )
            raw_paths.append(
                copy_local_file(
                    local_path=local_path,
                    output_path=output_path,
                    overwrite=overwrite,
                )
            )
            continue

        raise NotImplementedError(
            f"Unsupported generic raster download mode={mode!r}. "
            "Supported modes: manual, manual_url, local_file"
        )

    return raw_paths
