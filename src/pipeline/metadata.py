import json
from pathlib import Path


def build_feature_metadata(
    source_cfg: dict,
    variable: str,
    variable_cfg: dict,
    aggregation_cfg: dict,
    months: list[int],
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
    resampling_method_name: str,
) -> dict:
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    return {
        "source_provider": source["provider"],
        "source_product": source["product"],
        "source_version": str(source.get("version", "")),
        "source_description": source.get("description", ""),
        "source_period": source.get("source_period", ""),
        "source_crs": source.get("source_crs", ""),
        "source_resolution": processing["source_resolution"],
        "source_variable": variable,
        "source_unit": variable_cfg.get("unit", ""),
        "scale_factor": variable_cfg.get("scale_factor", 1.0),
        "clip_aoi": clip_aoi_name,
        "output_aoi": output_aoi_name,
        "target_crs": "EPSG:3035",
        "target_resolution_m": int(target_resolution_m),
        "months": months,
        "month_start": min(months),
        "month_end": max(months),
        "temporal_aggregation": aggregation_cfg["metric"],
        "aggregation_name": aggregation_cfg.get("name", ""),
        "resampling_method": resampling_method_name,
        "transformations": [
            "global_raw_zip_downloaded_or_provided",
            "bbox_clipped_to_intermediate_source_raster",
            "reprojected_to_project_grid",
            "resampled_to_target_resolution",
            "scale_factor_applied",
            "temporal_aggregation_applied",
        ],
        "important_note": (
            "Resampling aligns WorldClim data to the project grid. "
            "It does not increase the real climatic spatial resolution of the source data."
        ),
    }


def build_static_feature_metadata(
    source_cfg: dict,
    layer_name: str,
    layer_cfg: dict,
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
    resampling_method_name: str,
) -> dict:
    source = source_cfg["source"]
    processing = source_cfg["processing"]
    dataset = source_cfg.get("dataset", {})

    return {
        "source_provider": source["provider"],
        "source_id": source.get("id", ""),
        "source_product": source["product"],
        "source_product_group": source.get("product_group", ""),
        "source_version": str(source.get("version", "")),
        "source_description": source.get("description", ""),
        "source_page_url": source.get("page_url", ""),
        "source_documentation_url": source.get("documentation_url", ""),
        "source_base_url": source.get("base_url", ""),
        "source_period": source.get("source_period", ""),
        "source_citation": source.get("citation", ""),
        "source_crs": source.get("source_crs", ""),
        "source_resolution": processing["source_resolution"],
        "dataset_layer_structure": dataset.get("layer_structure", ""),
        "source_layer": layer_name,
        "source_layer_description": layer_cfg.get("description", ""),
        "source_unit": layer_cfg.get("unit", ""),
        "scale_factor": layer_cfg.get("scale_factor", 1.0),
        "clip_aoi": clip_aoi_name,
        "output_aoi": output_aoi_name,
        "target_crs": "EPSG:3035",
        "target_resolution_m": int(target_resolution_m),
        "resampling_method": resampling_method_name,
        "transformations": [
            "global_raw_zip_downloaded_or_provided",
            "bbox_clipped_to_intermediate_source_raster",
            "reprojected_to_project_grid",
            "resampled_to_target_resolution",
            "scale_factor_applied",
        ],
        "important_note": (
            "Resampling aligns WorldClim data to the project grid. "
            "It does not increase the real spatial resolution of the source data."
        ),
    }

def write_sidecar_json(metadata: dict, output_tif_path: Path) -> Path:
    output_json_path = output_tif_path.with_suffix(".json")

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return output_json_path


def metadata_to_geotiff_tags(metadata: dict) -> dict:
    """
    GeoTIFF tags must be simple string-like values.
    """
    tags = {}

    for key, value in metadata.items():
        tag_key = key.upper()

        if isinstance(value, list):
            tags[tag_key] = ",".join(str(v) for v in value)
        elif isinstance(value, dict):
            tags[tag_key] = json.dumps(value, ensure_ascii=False)
        else:
            tags[tag_key] = str(value)

    return tags