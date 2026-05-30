from __future__ import annotations

from pathlib import Path


SUPPORTED_LAYER_STRUCTURES = {
    "static_single",
    "static_multi",
    "yearly_static_collection",
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


def validate_generic_raster_source_config(source_cfg: dict, provider: str | None = None) -> None:
    source = source_cfg.get("source", {})
    processing = source_cfg.get("processing", {})

    if provider is not None and source.get("provider") != provider:
        raise ValueError(
            f"Expected provider={provider!r}, got provider={source.get('provider')!r}"
        )

    layer_structure = get_layer_structure(source_cfg)
    if layer_structure not in SUPPORTED_LAYER_STRUCTURES:
        raise NotImplementedError(
            f"Unsupported generic raster layer_structure={layer_structure!r}. "
            f"Supported: {sorted(SUPPORTED_LAYER_STRUCTURES)}"
        )

    file_format = get_file_format(source_cfg)
    if file_format not in SUPPORTED_FILE_FORMATS:
        raise NotImplementedError(
            f"Unsupported generic raster file_format={file_format!r}. "
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
    variables = source_cfg.get("variables", {}) or {}
    return [
        (variable, cfg)
        for variable, cfg in variables.items()
        if cfg.get("enabled", False)
    ]


def get_download_file_specs(source_cfg: dict) -> list[dict]:
    download_cfg = source_cfg.get("download", {}) or {}
    files_cfg = download_cfg.get("files", {}) or {}

    if files_cfg:
        return [
            {
                "variable": download_name,
                "filename": file_cfg.get("filename") or f"{download_name}.tif",
                "url": file_cfg.get("url"),
                "urls": file_cfg.get("urls"),
                "filenames": file_cfg.get("filenames"),
                "local_path": file_cfg.get("local_path"),
                "zip_member": file_cfg.get("zip_member"),
                "zip_member_pattern": file_cfg.get("zip_member_pattern"),
                "file_pattern": file_cfg.get("file_pattern"),
                "allow_multiple": bool(file_cfg.get("allow_multiple", False)),
                "allow_multiple_zip_members": bool(
                    file_cfg.get("allow_multiple_zip_members", False)
                ),
                "skip_zip_without_matching_members": bool(
                    file_cfg.get("skip_zip_without_matching_members", False)
                ),
                "postprocess": file_cfg.get("postprocess"),
            }
            for download_name, file_cfg in files_cfg.items()
        ]

    specs: list[dict] = []
    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        context = _format_context(source_cfg, variable_cfg)
        specs.append(
            {
                "variable": variable,
                "filename": _format_template(
                    variable_cfg.get("source_filename") or f"{variable}.tif",
                    context,
                ),
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


def get_file_spec_for_variable(source_cfg: dict, variable: str) -> dict:
    for spec in get_download_file_specs(source_cfg):
        if spec["variable"] == variable:
            return spec

    variable_cfg = source_cfg.get("variables", {}).get(variable, {}) or {}
    context = _format_context(source_cfg, variable_cfg)
    return {
        "variable": variable,
        "filename": _format_template(
            variable_cfg.get("source_filename") or f"{variable}.tif",
            context,
        ),
        "zip_member": _format_template(variable_cfg.get("zip_member"), context),
        "zip_member_pattern": _format_template(
            variable_cfg.get("zip_member_pattern"),
            context,
        ),
    }


def _format_context(source_cfg: dict, variable_cfg: dict) -> dict:
    context = dict(variable_cfg.get("generation_context", {}) or {})
    context.update(
        {
            "source_resolution": get_source_resolution(source_cfg),
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


def build_raw_path(raw_dir: Path, source_cfg: dict, variable: str) -> Path:
    spec = get_file_spec_for_variable(source_cfg, variable)
    return raw_dir / spec["filename"]


def _provider_prefix(source_cfg: dict) -> str:
    return str(source_cfg["source"]["provider"])


def build_clipped_name(source_cfg: dict, variable: str, domain_name: str) -> str:
    source = source_cfg["source"]
    source_resolution = get_source_resolution(source_cfg)
    return (
        f"{_provider_prefix(source_cfg)}_{source['product']}_{variable}_"
        f"{domain_name}_{source_resolution}_clipped.tif"
    )


def build_feature_name(
    source_cfg: dict,
    variable: str,
    domain_name: str,
    target_resolution_m: int,
) -> str:
    source = source_cfg["source"]
    return (
        f"{_provider_prefix(source_cfg)}_{source['product']}_{variable}_"
        f"{domain_name}_{target_resolution_m}m.tif"
    )
