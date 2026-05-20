from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io.config import get_repo_root, load_yaml, resolve_path
from src.pipeline.variable_expansion import expand_source_config
from src.workbench.temporal import infer_temporal_capability


SUPPORTED_METRICS = ["mean", "sum", "std", "min", "max"]
SUPPORTED_RESAMPLING = [
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]
SUPPORTED_STAGES = ["download", "clip", "build", "all"]
WORLDCLIM_SOURCE_RESOLUTIONS = ["30s", "2.5m", "5m", "10m"]


def _rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path)


def _clean(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel_path(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _resolution_unit(source_cfg: dict[str, Any]) -> str | None:
    dataset_cfg = source_cfg.get("dataset", {}) or {}
    processing_cfg = source_cfg.get("processing", {}) or {}
    source_cfg_meta = source_cfg.get("source", {}) or {}

    if dataset_cfg.get("native_resolution_m") is not None:
        return "metre"

    source_resolution = str(processing_cfg.get("source_resolution", ""))
    provider = source_cfg_meta.get("provider")

    if provider == "worldclim":
        if source_resolution.endswith("s"):
            return "arc_second"
        if source_resolution.endswith("m"):
            return "arc_minute"

    if source_resolution.endswith("m"):
        return "metre"

    return None


def _variable_items(source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for collection_name, kind in [("variables", "variable"), ("indices", "index")]:
        collection = source_cfg.get(collection_name, {}) or {}
        for name, cfg in collection.items():
            item = {
                "name": name,
                "kind": kind,
                "enabled_default": bool(cfg.get("enabled", False)),
                "description": cfg.get("description"),
                "unit": cfg.get("unit"),
                "scale_factor": cfg.get("scale_factor", 1.0),
                "valid_range": cfg.get("valid_range"),
                "data_type": cfg.get("data_type"),
                "native_resolution_m": cfg.get("native_resolution_m"),
                "index": cfg.get("index"),
                "resampling": cfg.get("resampling"),
                "temporal": cfg.get("temporal"),
                "generated_from": cfg.get("generated_from"),
            }
            items.append({key: value for key, value in item.items() if value is not None})

    return items


def _vector_layer_items(source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for dataset_name, dataset_cfg in (source_cfg.get("datasets", {}) or {}).items():
        for layer_name, layer_cfg in (dataset_cfg.get("layers", {}) or {}).items():
            item = {
                "name": f"{dataset_name}.{layer_name}",
                "dataset": dataset_name,
                "layer": layer_name,
                "kind": "vector_layer",
                "enabled_default": bool(
                    dataset_cfg.get("enabled", True)
                    and layer_cfg.get("enabled", True)
                ),
                "description": layer_cfg.get("description"),
                "geometry_type": layer_cfg.get("geometry_type"),
                "value_field": layer_cfg.get("value_field"),
            }
            items.append({key: value for key, value in item.items() if value is not None})

    return items


def _dimensions(source_cfg: dict[str, Any]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for key in ["gcms", "ssps", "periods", "years"]:
        value = source_cfg.get(key)
        if value is not None:
            dimensions[key] = value
    return dimensions


def _source_resolution_options(source_cfg: dict[str, Any]) -> list[str]:
    processing_cfg = source_cfg.get("processing", {}) or {}
    source_cfg_meta = source_cfg.get("source", {}) or {}

    configured = processing_cfg.get("source_resolutions")
    if isinstance(configured, list):
        return [str(item) for item in configured]

    source_resolution = processing_cfg.get("source_resolution")
    provider = source_cfg_meta.get("provider")

    if provider == "worldclim":
        return WORLDCLIM_SOURCE_RESOLUTIONS

    return [str(source_resolution)] if source_resolution is not None else []


def _keep_raw_after_clip_default(source_cfg: dict[str, Any]) -> bool:
    download_cfg = source_cfg.get("download", {}) or {}

    if "keep_raw_after_clip" in download_cfg:
        return bool(download_cfg["keep_raw_after_clip"])

    if "delete_raw_after_clip" in download_cfg:
        return not bool(download_cfg["delete_raw_after_clip"])

    for key in [
        "keep_global_file_after_clip",
        "keep_global_zip_after_clip",
        "keep_raw_zip_after_clip",
    ]:
        if key in download_cfg:
            return bool(download_cfg[key])

    return True


def source_catalog_from_config(
    source_config_path: str | Path,
) -> dict[str, Any]:
    path = resolve_path(source_config_path, must_exist=True)
    cfg = load_yaml(path)
    expanded_cfg = expand_source_config(cfg)

    source = cfg.get("source", {}) or {}
    dataset = cfg.get("dataset", {}) or {}
    processing = cfg.get("processing", {}) or {}

    source_resolution = processing.get("source_resolution")
    native_resolution = dataset.get("native_resolution", source_resolution)

    catalog = {
        "id": source.get("id") or path.stem,
        "provider": source.get("provider"),
        "product": source.get("product"),
        "product_group": source.get("product_group"),
        "version": source.get("version"),
        "description": source.get("description"),
        "config_path": _rel_path(path),
        "source_crs": source.get("source_crs") or dataset.get("source_crs"),
        "source_period": source.get("source_period"),
        "source_scale": source.get("source_scale"),
        "native_resolution": native_resolution,
        "native_resolution_m": dataset.get("native_resolution_m"),
        "native_resolution_unit": _resolution_unit(cfg),
        "source_resolution": source_resolution,
        "source_resolution_options": _source_resolution_options(cfg),
        "target_resolution_m": processing.get("target_resolution_m"),
        "keep_raw_after_clip_default": _keep_raw_after_clip_default(cfg),
        "layer_structure": dataset.get("layer_structure"),
        "file_format": dataset.get("file_format"),
        "data_type": dataset.get("data_type"),
        "variables": _variable_items(expanded_cfg),
        "layers": _vector_layer_items(expanded_cfg),
        "dimensions": _dimensions(expanded_cfg),
        "aggregations": expanded_cfg.get("temporal_aggregations", []) or [],
        "temporal": infer_temporal_capability(expanded_cfg),
        "resampling": expanded_cfg.get("resampling", {}) or {},
        "citation": source.get("citation"),
        "page_url": source.get("page_url"),
        "doi": source.get("doi"),
    }

    return _clean(
        {key: value for key, value in catalog.items() if value not in [None, [], {}]}
    )


def list_source_catalogs() -> list[dict[str, Any]]:
    source_root = get_repo_root() / "configs" / "sources"
    catalogs = [
        source_catalog_from_config(path)
        for path in sorted(source_root.rglob("*.yaml"))
    ]
    return sorted(catalogs, key=lambda item: item.get("id", ""))


def list_aoi_catalogs() -> list[dict[str, Any]]:
    aoi_root = get_repo_root() / "configs" / "aoi"
    aois: list[dict[str, Any]] = []

    for path in sorted(aoi_root.glob("*.yaml")):
        cfg = load_yaml(path)
        aois.append(
            _clean(
                {
                    "name": cfg.get("name") or cfg.get("aoi", {}).get("name") or path.stem,
                    "path": path,
                    "description": cfg.get("description"),
                    "crs": cfg.get("crs"),
                    "bounds": cfg.get("bounds"),
                }
            )
        )

    return aois


def project_catalog(
    project_config_path: str | Path = "configs/project.yaml",
) -> dict[str, Any]:
    path = resolve_path(project_config_path, must_exist=True)
    cfg = load_yaml(path)

    grids = cfg.get("grids", {}) or {}

    return _clean(
        {
            "config_path": path,
            "name": cfg.get("project", {}).get("name", "pirineus-raster"),
            "crs": cfg.get("crs"),
            "nodata": cfg.get("nodata"),
            "available_resolutions_m": grids.get("available_resolutions_m", []),
            "default_resolution_m": grids.get("default_resolution_m"),
            "paths": cfg.get("paths", {}),
        }
    )


def workbench_catalog(
    project_config_path: str | Path = "configs/project.yaml",
) -> dict[str, Any]:
    return {
        "project": project_catalog(project_config_path),
        "aois": list_aoi_catalogs(),
        "sources": list_source_catalogs(),
        "supported_metrics": SUPPORTED_METRICS,
        "supported_resampling": SUPPORTED_RESAMPLING,
        "supported_stages": SUPPORTED_STAGES,
    }
