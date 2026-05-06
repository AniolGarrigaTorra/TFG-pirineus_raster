from __future__ import annotations

from pathlib import Path


SUPPORTED_LAYER_STRUCTURES = {
    "static_single",
    "static_multi",
}


SUPPORTED_FILE_FORMATS = {
    "geotiff",
    "zip_geotiff",
}


def get_layer_structure(source_cfg: dict) -> str:
    return source_cfg.get("dataset", {}).get("layer_structure", "static_single")


def get_file_format(source_cfg: dict) -> str:
    return source_cfg.get("dataset", {}).get("file_format", "geotiff")


def get_source_resolution(source_cfg: dict) -> str:
    return str(source_cfg["processing"]["source_resolution"])


def validate_copernicus_source_config(source_cfg: dict) -> None:
    source = source_cfg.get("source", {})
    dataset = source_cfg.get("dataset", {})
    processing = source_cfg.get("processing", {})

    if source.get("provider") != "copernicus":
        raise ValueError(
            f"Expected provider='copernicus', got provider={source.get('provider')!r}"
        )

    layer_structure = get_layer_structure(source_cfg)
    if layer_structure not in SUPPORTED_LAYER_STRUCTURES:
        raise NotImplementedError(
            f"Unsupported Copernicus layer_structure={layer_structure!r}. "
            f"Supported: {sorted(SUPPORTED_LAYER_STRUCTURES)}"
        )

    file_format = get_file_format(source_cfg)
    if file_format not in SUPPORTED_FILE_FORMATS:
        raise NotImplementedError(
            f"Unsupported Copernicus file_format={file_format!r}. "
            f"Supported: {sorted(SUPPORTED_FILE_FORMATS)}"
        )

    if "source_resolution" not in processing:
        raise ValueError("Missing processing.source_resolution")

    if "target_resolution_m" not in processing:
        raise ValueError("Missing processing.target_resolution_m")

    variables = source_cfg.get("variables", {})
    enabled = [
        variable
        for variable, cfg in variables.items()
        if cfg.get("enabled", False)
    ]

    if not enabled:
        raise ValueError("No enabled variables found in source config.")


def get_enabled_variable_items(source_cfg: dict) -> list[tuple[str, dict]]:
    variables = source_cfg.get("variables", {})

    return [
        (variable, cfg)
        for variable, cfg in variables.items()
        if cfg.get("enabled", False)
    ]


def get_download_file_specs(source_cfg: dict) -> list[dict]:
    """
    Return one file spec per enabled variable.

    Supported patterns:

    1) Direct URL single file:
      download:
        files:
          variable:
            url: "..."
            filename: "variable.tif"

    2) Direct URL multiple files -> mosaic:
      download:
        files:
          variable:
            filename: "variable_mosaic.tif"
            urls:
              - "https://..."
              - "https://..."
            postprocess: mosaic_geotiff

    3) WEkEO HDA tiled ZIPs -> raw GeoTIFF mosaic:
      download:
        mode: wekeo_hda
        files:
          variable:
            filename: "variable_mosaic.tif"
            file_pattern: "*.zip"
            zip_member_pattern: ".*\\.tif$"
            postprocess: mosaic_zip_geotiff
            hda_query:
              dataset_id: "..."
    """
    download_cfg = source_cfg.get("download", {}) or {}
    files_cfg = download_cfg.get("files", {}) or {}

    specs: list[dict] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        file_cfg = files_cfg.get(variable, {}) or {}

        filename = (
            file_cfg.get("filename")
            or variable_cfg.get("source_filename")
            or f"{variable}.tif"
        )

        spec = {
            "variable": variable,
            "filename": filename,
            "url": file_cfg.get("url") or variable_cfg.get("url"),
            "urls": file_cfg.get("urls") or variable_cfg.get("urls"),
            "filenames": file_cfg.get("filenames") or variable_cfg.get("filenames"),
            "local_path": file_cfg.get("local_path") or variable_cfg.get("local_path"),
            "zip_member": file_cfg.get("zip_member") or variable_cfg.get("zip_member"),
            "zip_member_pattern": (
                file_cfg.get("zip_member_pattern")
                or variable_cfg.get("zip_member_pattern")
            ),
            "hda_query": file_cfg.get("hda_query") or variable_cfg.get("hda_query"),
            "hda_query_path": (
                file_cfg.get("hda_query_path")
                or variable_cfg.get("hda_query_path")
            ),
            "file_pattern": (
                file_cfg.get("file_pattern")
                or variable_cfg.get("file_pattern")
            ),
            "max_results": (
                file_cfg.get("max_results")
                or variable_cfg.get("max_results")
            ),
            "allow_multiple": bool(
                file_cfg.get(
                    "allow_multiple",
                    variable_cfg.get("allow_multiple", False),
                )
            ),
            "postprocess": (
                file_cfg.get("postprocess")
                or variable_cfg.get("postprocess")
            ),
            "allow_multiple_zip_members": bool(
                file_cfg.get(
                    "allow_multiple_zip_members",
                    variable_cfg.get("allow_multiple_zip_members", False),
                )
            ),
            "skip_zip_without_matching_members": bool(
                file_cfg.get(
                    "skip_zip_without_matching_members",
                    variable_cfg.get("skip_zip_without_matching_members", False),
                )
            ),
        }

        specs.append(spec)

    return specs


def get_file_spec_for_variable(source_cfg: dict, variable: str) -> dict:
    for spec in get_download_file_specs(source_cfg):
        if spec["variable"] == variable:
            return spec

    raise KeyError(f"No file spec found for variable={variable!r}")


def build_copernicus_raw_path(
    raw_dir: Path,
    source_cfg: dict,
    variable: str,
) -> Path:
    spec = get_file_spec_for_variable(source_cfg, variable)
    return raw_dir / spec["filename"]


def build_copernicus_clipped_name(
    source_cfg: dict,
    variable: str,
    domain_name: str,
) -> str:
    source = source_cfg["source"]
    processing = source_cfg["processing"]

    product = source["product"]
    source_resolution = processing["source_resolution"]

    return (
        f"copernicus_{product}_{variable}_"
        f"{domain_name}_{source_resolution}_clipped.tif"
    )


def build_copernicus_feature_name(
    source_cfg: dict,
    variable: str,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    source = source_cfg["source"]
    product = source["product"]

    return (
        f"copernicus_{product}_{variable}_"
        f"{domain_name}_{target_resolution_m}m.tif"
    )