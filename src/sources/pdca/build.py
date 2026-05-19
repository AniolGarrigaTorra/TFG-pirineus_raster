from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rasterio

from src.io.paths import ensure_dir, get_feature_output_dir, get_source_interim_dir
from src.pipeline.raster_ops import (
    get_variable_resampling_method,
    get_variable_resampling_method_name,
    load_grid_context,
    print_grid_context,
    read_raster_to_grid,
    write_feature_raster,
)
from src.sources.pdca.naming import feature_raster_name, variable_key_from_layer_id


def _output_options(project_cfg: dict, source_cfg: dict) -> dict:
    output_cfg = source_cfg.get("output", {})
    return {
        "nodata": float(project_cfg.get("nodata", -9999.0)),
        "output_dtype": output_cfg.get("dtype", "float32"),
        "compression": output_cfg.get("compression", "LZW"),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _target_resolution(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


def _source_metadata(source_cfg: dict) -> dict[str, Any]:
    source = source_cfg["source"]
    dataset = source_cfg.get("dataset", {})
    return {
        "provider": source.get("provider"),
        "product": source.get("product"),
        "source_id": source.get("id"),
        "source_config_path": source_cfg.get("_config_path"),
        "version": source.get("version"),
        "description": source.get("description"),
        "citation": source.get("citation"),
        "article_url": source.get("article_url"),
        "doi": source.get("doi"),
        "source_period": source.get("source_period"),
        "source_resolution": source_cfg.get("processing", {}).get("source_resolution"),
        "native_resolution": (
            dataset.get("native_resolution")
            or source_cfg.get("processing", {}).get("source_resolution")
        ),
        "native_resolution_m": dataset.get("native_resolution_m"),
        "source_crs": source.get("source_crs") or dataset.get("source_crs"),
    }


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, tuple):
            cleaned[key] = list(value)
        else:
            cleaned[key] = value
    return cleaned


def _variable_cfg(source_cfg: dict, variable_key: str) -> dict:
    return source_cfg.get("variables", {}).get(variable_key, {})


def _scale_factor(variable_cfg: dict) -> float:
    return float(variable_cfg.get("scale_factor", 1.0))


def _list_clipped_paths(project_cfg: dict, source_cfg: dict, clip_aoi_name: str) -> list[Path]:
    source = source_cfg["source"]
    source_resolution = source_cfg["processing"]["source_resolution"]
    clipped_root = (
        get_source_interim_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
        )
        / "clipped"
        / clip_aoi_name
        / source_resolution
    )
    if not clipped_root.exists():
        raise FileNotFoundError(f"PDCA clipped root does not exist: {clipped_root}")

    paths = sorted(path for path in clipped_root.rglob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No clipped PDCA GeoTIFFs found in {clipped_root}")
    return paths


def _tags_lower(tags: dict[str, str]) -> dict[str, str]:
    return {str(k).lower(): v for k, v in tags.items()}


def build_pdca_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    source = source_cfg["source"]
    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _target_resolution(source_cfg)
    output_options = _output_options(project_cfg, source_cfg)

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    print(f"[pdca:build] Output AOI: {output_aoi_name}")
    print(f"[pdca:build] Clip AOI: {clip_aoi_name}")
    print_grid_context(grid, prefix="[pdca:build]")

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )
    ensure_dir(output_dir)

    clipped_paths = _list_clipped_paths(project_cfg, source_cfg, clip_aoi_name)
    print(f"[pdca:build] Clipped rasters: {len(clipped_paths)}")

    written_paths: list[Path] = []
    manifest: list[dict[str, Any]] = []

    for clipped_path in clipped_paths:
        layer_id = clipped_path.parent.name
        with rasterio.open(clipped_path) as src:
            source_tags = _tags_lower(src.tags())

        variable_key = variable_key_from_layer_id(layer_id, source_tags)
        var_cfg = _variable_cfg(source_cfg, variable_key)
        resampling = get_variable_resampling_method(source_cfg, variable_key)
        resampling_name = get_variable_resampling_method_name(source_cfg, variable_key)
        scale_factor = _scale_factor(var_cfg)

        output_path = output_dir / feature_raster_name(
            provider=source["provider"],
            product=source["product"],
            layer_id=layer_id,
            domain_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
        )

        print("==============================")
        print(f"[pdca:build] Layer: {layer_id}")
        print(f"[pdca:build] Variable key: {variable_key}")
        print(f"[pdca:build] Temporal kind: {source_tags.get('temporal_kind')}")
        print(f"[pdca:build] Period: {source_tags.get('period')}")
        print(f"[pdca:build] Resampling: {resampling_name}")
        print(f"[pdca:build] Input: {clipped_path}")
        print(f"[pdca:build] Output: {output_path}")

        array = read_raster_to_grid(
            raster_path=clipped_path,
            grid=grid,
            resampling=resampling,
            band=1,
            scale_factor=scale_factor,
        )

        metadata = _clean_metadata(
            {
                **_source_metadata(source_cfg),
                "layer_id": layer_id,
                "variable": variable_key,
                "variable_description": var_cfg.get("description"),
                "unit": var_cfg.get("unit"),
                "valid_range": var_cfg.get("valid_range"),
                "scale_factor": scale_factor,
                "temporal_kind": source_tags.get("temporal_kind"),
                "period": source_tags.get("period"),
                "archive_stem": source_tags.get("archive_stem"),
                "canonical_prefix": source_tags.get("canonical_prefix"),
                "clip_aoi_name": clip_aoi_name,
                "output_aoi_name": output_aoi_name,
                "target_resolution_m": target_resolution_m,
                "resampling": resampling_name,
                "source_clipped_path": str(clipped_path),
                "source_raster": source_tags.get("source_raster"),
            }
        )

        written = write_feature_raster(
            output_path=output_path,
            array=array,
            grid=grid,
            metadata=metadata,
            **output_options,
            validate=True,
        )
        written_paths.append(written)
        manifest.append(
            {
                "layer_id": layer_id,
                "variable": variable_key,
                "temporal_kind": source_tags.get("temporal_kind"),
                "period": source_tags.get("period"),
                "input": str(clipped_path),
                "output": str(written),
                "resampling": resampling_name,
            }
        )

    manifest_path = output_dir / "pdca_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[pdca:build] Manifest: {manifest_path}")

    return written_paths
