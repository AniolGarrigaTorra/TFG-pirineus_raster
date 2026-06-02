from __future__ import annotations

from pathlib import Path


SUPPORTED_LAYER_STRUCTURES = {
    "static_single",
    "static_multi",
    "yearly_static_collection",
    "temporal_aggregation",
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


def get_source_resolution_token(source_cfg: dict) -> str:
    processing = source_cfg.get("processing", {}) or {}
    source_resolution = get_source_resolution(source_cfg)
    tokens = processing.get("source_resolution_tokens", {}) or {}
    return str(tokens.get(source_resolution, source_resolution))


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
    Return download file specs.

    Important design:

    - If download.files exists, it is the authoritative list of download tasks.
      This is needed for temporal products where one download task can generate
      many output variables.

      Example:
        download.files.snow_scenes
          -> downloads many HRSI snow scenes
          -> temporal_postprocess generates:
               snow_fraction_mean_winter
               snow_cover_days_winter
               ...

    - If download.files does not exist, fallback to one download spec per
      enabled variable. This keeps old/simple static source configs working.
    """
    download_cfg = source_cfg.get("download", {}) or {}
    files_cfg = download_cfg.get("files", {}) or {}

    specs: list[dict] = []

    # ------------------------------------------------------------
    # Preferred path: explicit download.files
    # ------------------------------------------------------------
    if files_cfg:
        for download_name, file_cfg in files_cfg.items():
            file_cfg = file_cfg or {}

            filename = (
                file_cfg.get("filename")
                or f"{download_name}.tif"
            )

            spec = {
                "variable": download_name,
                "filename": filename,

                # Direct/manual URL modes
                "url": file_cfg.get("url"),
                "urls": file_cfg.get("urls"),
                "filenames": file_cfg.get("filenames"),
                "local_path": file_cfg.get("local_path"),

                # ZIP handling
                "zip_member": file_cfg.get("zip_member"),
                "zip_member_pattern": file_cfg.get("zip_member_pattern"),
                "file_pattern": file_cfg.get("file_pattern"),

                # WEkEO HDA
                "hda_query": file_cfg.get("hda_query"),
                "hda_query_path": file_cfg.get("hda_query_path"),
                "max_results": file_cfg.get("max_results"),

                # Multi-file/multi-member behavior
                "allow_multiple": bool(file_cfg.get("allow_multiple", False)),
                "allow_multiple_zip_members": bool(
                    file_cfg.get("allow_multiple_zip_members", False)
                ),
                "skip_zip_without_matching_members": bool(
                    file_cfg.get("skip_zip_without_matching_members", False)
                ),

                # Processing mode
                "postprocess": file_cfg.get("postprocess"),
            }

            specs.append(spec)

        return specs

    # ------------------------------------------------------------
    # Fallback path: one spec per enabled variable
    # ------------------------------------------------------------
    # This supports older/simple source configs where each variable corresponds
    # directly to one raw input file.
    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        context = _format_context(source_cfg, variable_cfg)
        filename = variable_cfg.get("source_filename") or f"{variable}.tif"

        specs.append(
            {
                "variable": variable,
                "filename": _format_template(filename, context),

                "url": _format_template(variable_cfg.get("url"), context),
                "urls": _format_template(variable_cfg.get("urls"), context),
                "filenames": _format_template(variable_cfg.get("filenames"), context),
                "local_path": _format_template(variable_cfg.get("local_path"), context),

                "zip_member": _format_template(variable_cfg.get("zip_member"), context),
                "zip_member_pattern": _format_template(
                    variable_cfg.get("zip_member_pattern"),
                    context,
                ),
                "file_pattern": _format_template(variable_cfg.get("file_pattern"), context),

                "hda_query": _format_template(variable_cfg.get("hda_query"), context),
                "hda_query_path": _format_template(
                    variable_cfg.get("hda_query_path"),
                    context,
                ),
                "max_results": variable_cfg.get("max_results"),

                "allow_multiple": bool(variable_cfg.get("allow_multiple", False)),
                "allow_multiple_zip_members": bool(
                    variable_cfg.get("allow_multiple_zip_members", False)
                ),
                "skip_zip_without_matching_members": bool(
                    variable_cfg.get("skip_zip_without_matching_members", False)
                ),

                "postprocess": variable_cfg.get("postprocess"),
            }
        )

    return specs


def _format_context(source_cfg: dict, variable_cfg: dict) -> dict:
    context = dict(variable_cfg.get("generation_context", {}) or {})
    context.update(
        {
            "source_resolution": get_source_resolution(source_cfg),
            "source_resolution_token": get_source_resolution_token(source_cfg),
            "target_resolution_m": source_cfg.get("processing", {}).get("target_resolution_m"),
        }
    )
    return context


def _format_template(value, context: dict):
    if value is None:
        return None
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_format_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_template(item, context) for key, item in value.items()}
    return value


def get_file_spec_for_variable(source_cfg: dict, variable: str) -> dict:
    for spec in get_download_file_specs(source_cfg):
        if spec["variable"] == variable:
            return spec

    variable_cfg = source_cfg.get("variables", {}).get(variable, {}) or {}
    context = _format_context(source_cfg, variable_cfg)
    filename = variable_cfg.get("source_filename") or f"{variable}.tif"

    return {
        "variable": variable,
        "filename": _format_template(filename, context),
        "zip_member": _format_template(variable_cfg.get("zip_member"), context),
        "zip_member_pattern": _format_template(
            variable_cfg.get("zip_member_pattern"),
            context,
        ),
    }


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
