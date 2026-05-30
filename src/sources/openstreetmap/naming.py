from __future__ import annotations

from pathlib import Path


def get_source_resolution(source_cfg: dict) -> str:
    return str(source_cfg.get("processing", {}).get("source_resolution", "osm"))


def get_enabled_layer_items(source_cfg: dict) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    for dataset_name, dataset_cfg in (source_cfg.get("datasets", {}) or {}).items():
        if not bool(dataset_cfg.get("enabled", True)):
            continue
        for layer_name, layer_cfg in (dataset_cfg.get("layers", {}) or {}).items():
            if bool(layer_cfg.get("enabled", True)):
                items.append((f"{dataset_name}.{layer_name}", layer_cfg))
    return items


def validate_osm_source_config(source_cfg: dict) -> None:
    source = source_cfg.get("source", {})
    if source.get("provider") != "openstreetmap":
        raise ValueError(
            "Expected provider='openstreetmap', "
            f"got provider={source.get('provider')!r}"
        )
    if not get_enabled_layer_items(source_cfg):
        raise ValueError("No enabled OpenStreetMap layers found in source config.")
    processing = source_cfg.get("processing", {}) or {}
    if "target_resolution_m" not in processing:
        raise ValueError("Missing processing.target_resolution_m")


def build_osm_raw_name(region_name: str) -> str:
    safe = str(region_name).replace("/", "_").replace(" ", "_")
    if safe.endswith(".osm.pbf"):
        return safe
    return f"{safe}.osm.pbf"


def build_osm_clipped_name(layer_key: str, domain_name: str) -> str:
    safe_layer = layer_key.replace(".", "_")
    return f"openstreetmap_geofabrik_{safe_layer}_{domain_name}_clipped.gpkg"


def build_osm_feature_name(layer_key: str, domain_name: str, target_resolution_m: int) -> str:
    safe_layer = layer_key.replace(".", "_")
    return f"openstreetmap_geofabrik_{safe_layer}_{domain_name}_{target_resolution_m}m.tif"


def build_osm_raw_path(raw_dir: Path, region_cfg: dict) -> Path:
    return raw_dir / str(region_cfg.get("filename") or build_osm_raw_name(region_cfg["name"]))
