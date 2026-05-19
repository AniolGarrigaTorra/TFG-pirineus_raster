from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from src.io.paths import ensure_dir, get_source_clipped_dir, get_source_interim_dir, get_source_raw_dir
from src.sources.pdca.naming import (
    PDCA_VARIABLES,
    canonical_layer_id,
    clipped_raster_name,
    find_single_raster,
    safe_slug,
    strip_pyrenees_suffix,
    temporal_kind,
)


def _aoi_bounds(aoi_cfg: dict) -> tuple[float, float, float, float]:
    b = aoi_cfg["bounds"]
    return float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])


def _transform_bounds(
    bounds: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform_bounds(*bounds, densify_pts=21)


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            resolved = (target_dir / member.filename).resolve()
            if not str(resolved).startswith(str(target_root)):
                raise RuntimeError(f"Unsafe ZIP member in {zip_path}: {member.filename}")
        zf.extractall(target_dir)


def _extract_zip_once(zip_path: Path, target_dir: Path, overwrite: bool) -> None:
    marker = target_dir / ".extracted_ok"
    if marker.exists() and not overwrite:
        return
    print(f"[pdca:clip] Extracting: {zip_path} -> {target_dir}")
    _safe_extract_zip(zip_path, target_dir)
    marker.write_text(str(zip_path), encoding="utf-8")


def _expected_dataset(source_cfg: dict) -> dict[str, dict[str, Any]]:
    """Return the explicit PDCA structure.

    The default mirrors Zenodo record 1186639:
      - 7 variables with Annual + 12 months + 4 seasons = 119 layers
      - GDD with one layer = 1 layer
      - total = 120 layers
    The YAML may override this through dataset.expected_variables.
    """
    expected = source_cfg.get("dataset", {}).get("expected_variables")
    if not expected:
        return PDCA_VARIABLES

    parsed: dict[str, dict[str, Any]] = {}
    for item in expected:
        archive_stem = item["archive_stem"]
        parsed[archive_stem] = {
            "variable_key": item["variable_key"],
            "canonical_prefix": item["canonical_prefix"],
            "temporal_layers": item.get("temporal_layers", []),
        }
    return parsed


def _top_zip_path(raw_dir: Path, archive_stem: str) -> Path:
    path = raw_dir / f"{archive_stem}.zip"
    if not path.exists():
        raise FileNotFoundError(f"Expected PDCA archive not found: {path}")
    return path


def _inner_zip_path(top_extract_dir: Path, archive_stem: str, period: str | None) -> Path:
    # Top-level ZIP extraction creates: _extracted/Temperature_min/Temperature_min/...
    base_dir = top_extract_dir / archive_stem
    if period is None:
        return base_dir / f"{archive_stem}.zip"
    return base_dir / f"{archive_stem}_{period}.zip"


def _extracted_layer_dir(top_extract_dir: Path, archive_stem: str, period: str | None) -> Path:
    base_dir = top_extract_dir / archive_stem
    if period is None:
        return base_dir / archive_stem
    return base_dir / f"{archive_stem}_{period}"


def _extract_expected_archives(
    raw_dir: Path,
    extracted_root: Path,
    expected: dict[str, dict[str, Any]],
    overwrite: bool,
) -> None:
    ensure_dir(extracted_root)

    for archive_stem, spec in expected.items():
        top_zip = _top_zip_path(raw_dir, archive_stem)
        top_extract_dir = extracted_root / archive_stem
        _extract_zip_once(top_zip, top_extract_dir, overwrite=overwrite)

        for period in spec["temporal_layers"]:
            inner_zip = _inner_zip_path(top_extract_dir, archive_stem, period)
            if not inner_zip.exists():
                raise FileNotFoundError(f"Expected PDCA nested archive not found: {inner_zip}")
            layer_dir = _extracted_layer_dir(top_extract_dir, archive_stem, period)
            _extract_zip_once(inner_zip, layer_dir, overwrite=overwrite)


def _expected_layers(
    extracted_root: Path,
    expected: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for archive_stem, spec in expected.items():
        top_extract_dir = extracted_root / archive_stem
        for period in spec["temporal_layers"]:
            layer_dir = _extracted_layer_dir(top_extract_dir, archive_stem, period)
            raster_path = find_single_raster(layer_dir)
            layer_id = canonical_layer_id(spec["canonical_prefix"], period)
            # Some inner TIFFs include _pyrenees in the stem; keep the canonical id clean.
            layer_id = strip_pyrenees_suffix(layer_id)
            layers.append(
                {
                    "archive_stem": archive_stem,
                    "period": None if period is None else safe_slug(period),
                    "temporal_kind": temporal_kind(period),
                    "variable_key": spec["variable_key"],
                    "canonical_prefix": spec["canonical_prefix"],
                    "layer_id": layer_id,
                    "raster_path": raster_path,
                }
            )
    return layers


def _intersect_bounds(
    wanted: tuple[float, float, float, float],
    available: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    left = max(wanted[0], available[0])
    bottom = max(wanted[1], available[1])
    right = min(wanted[2], available[2])
    top = min(wanted[3], available[3])
    if left >= right or bottom >= top:
        return None
    return left, bottom, right, top


def _clip_one_raster(
    raster_path: Path,
    output_path: Path,
    clip_bounds_source_crs: tuple[float, float, float, float],
    compression: str,
    overwrite: bool,
    tags: dict[str, Any],
) -> bool:
    if output_path.exists() and not overwrite:
        print(f"[pdca:clip] Exists, skipping: {output_path}")
        return True

    ensure_dir(output_path.parent)

    with rasterio.open(raster_path) as src:
        source_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        read_bounds = _intersect_bounds(clip_bounds_source_crs, source_bounds)
        if read_bounds is None:
            print(f"[pdca:clip] No overlap with AOI, skipping: {raster_path}")
            return False

        window = from_bounds(*read_bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "compress": compression,
            }
        )
        for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
            profile.pop(key, None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                SOURCE_PROVIDER="pdca",
                SOURCE_RASTER=str(raster_path),
                CLIP_METHOD="bbox_intersection",
                CLIP_BOUNDS_SOURCE_CRS=str(clip_bounds_source_crs),
                SOURCE_BOUNDS=str(source_bounds),
                READ_BOUNDS=str(read_bounds),
                **{k.upper(): str(v) for k, v in tags.items() if v is not None},
            )

    print(f"[pdca:clip] Written: {output_path}")
    return True


def clip_pdca_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    clip_cfg = source_cfg.get("clip", {})

    provider = source["provider"]
    product = source["product"]
    source_resolution = processing["source_resolution"]
    source_crs = source.get("source_crs") or source_cfg.get("dataset", {}).get("source_crs")
    if not source_crs:
        raise ValueError("PDCA source_crs is required in source config.")

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=provider,
        product=product,
        source_resolution=source_resolution,
    )
    extracted_root = raw_dir / "_extracted"
    clip_aoi_name = clip_aoi_cfg["name"]
    project_crs = project_cfg["crs"]
    clip_bounds_source_crs = _transform_bounds(
        bounds=_aoi_bounds(clip_aoi_cfg),
        src_crs=project_crs,
        dst_crs=source_crs,
    )

    compression = source_cfg.get("output", {}).get("compression", "LZW")
    overwrite_clip = bool(clip_cfg.get("overwrite_existing", False))
    overwrite_extract = bool(clip_cfg.get("overwrite_extracted", False))
    expected = _expected_dataset(source_cfg)
    expected_count = sum(len(spec["temporal_layers"]) for spec in expected.values())

    print(f"[pdca:clip] Raw dir: {raw_dir}")
    print(f"[pdca:clip] Extracted dir: {extracted_root}")
    print(f"[pdca:clip] Clip AOI: {clip_aoi_name}")
    print(f"[pdca:clip] Project CRS: {project_crs}")
    print(f"[pdca:clip] Source CRS: {source_crs}")
    print(f"[pdca:clip] Clip bounds in source CRS: {clip_bounds_source_crs}")
    print(f"[pdca:clip] Expected PDCA layers: {expected_count}")

    _extract_expected_archives(raw_dir, extracted_root, expected, overwrite=overwrite_extract)
    layers = _expected_layers(extracted_root, expected)
    if len(layers) != expected_count:
        raise RuntimeError(f"Expected {expected_count} PDCA layers, found {len(layers)}")

    written_paths: list[Path] = []
    manifest: list[dict[str, Any]] = []

    for layer in layers:
        layer_id = layer["layer_id"]
        clipped_dir = get_source_clipped_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
            domain_name=clip_aoi_name,
            source_resolution=source_resolution,
            variable=layer_id,
        )
        output_path = clipped_dir / clipped_raster_name(layer_id, clip_aoi_name)
        before_exists = output_path.exists()

        ok = _clip_one_raster(
            raster_path=layer["raster_path"],
            output_path=output_path,
            clip_bounds_source_crs=clip_bounds_source_crs,
            compression=compression,
            overwrite=overwrite_clip,
            tags=layer,
        )
        if ok and output_path.exists():
            written_paths.append(output_path)
            manifest.append(
                {
                    **{k: str(v) if isinstance(v, Path) else v for k, v in layer.items()},
                    "clipped_path": str(output_path),
                    "already_existed": before_exists and not overwrite_clip,
                }
            )

    if len(written_paths) != expected_count:
        raise RuntimeError(f"Expected {expected_count} clipped PDCA files, wrote/found {len(written_paths)}")

    manifest_path = (
        get_source_interim_dir(
            project_cfg=project_cfg,
            provider=provider,
            product=product,
        )
        / "clipped"
        / clip_aoi_name
        / source_resolution
        / "pdca_clip_manifest.json"
    )
    ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[pdca:clip] Manifest: {manifest_path}")

    return written_paths
