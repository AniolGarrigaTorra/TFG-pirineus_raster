from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.pipeline.layer_spec import LayerSpec


def read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_range_from_metadata(metadata: dict[str, Any]) -> tuple[float, float] | None:
    value = metadata.get("valid_range")

    if value is None:
        value = metadata.get("variable_valid_range")

    if value is None:
        return None

    if isinstance(value, str):
        # Accept strings like "[-50, 60]" or "-50,60"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(numbers) >= 2:
            return (float(numbers[0]), float(numbers[1]))
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))

    return None


def _months_from_metadata(metadata: dict[str, Any]) -> list[int] | None:
    value = metadata.get("months")

    if value is None:
        value = metadata.get("aggregation_months")

    if value is None:
        return None

    if isinstance(value, list):
        return [int(v) for v in value]

    if isinstance(value, str):
        numbers = re.findall(r"\d+", value)
        if numbers:
            return [int(v) for v in numbers]

    return None


def _infer_variable_from_name(name: str) -> str | None:
    """
    Conservative filename fallback.

    Expected examples:
      worldclim_v2_1_elevation_elev_experimental_pallars_sobira_100m
      worldclim_v2_1_base_tmin_may_sep_mean_experimental_pallars_sobira_100m
    """
    known_variables = [
        "tmin",
        "tmax",
        "tavg",
        "prec",
        "srad",
        "wind",
        "vapr",
        "bio",
        "elev",
    ]

    tokens = name.split("_")
    for token in tokens:
        if token in known_variables:
            return token

    # Handle bio01, bio02, etc.
    for token in tokens:
        if re.fullmatch(r"bio\d{1,2}", token):
            return token

    return None


def _infer_resolution_from_name(name: str) -> int | None:
    match = re.search(r"_(\d+)m$", name)
    if match:
        return int(match.group(1))
    return None


def _metadata_first(metadata: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in metadata and metadata[key] not in [None, ""]:
            return metadata[key]
    return None


def build_layer_spec_from_raster_entry(
    raster_entry: dict[str, Any],
    source_result: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> LayerSpec:
    """
    Build one LayerSpec from a copied raster entry in the manifest.

    raster_entry comes from dataset_output.copy_raster_to_dataset().
    """
    source_result = source_result or {}
    manifest = manifest or {}

    dataset_path = Path(raster_entry["dataset_path"])
    original_path = Path(raster_entry["original_path"])

    sidecar_path_str = raster_entry.get("sidecar_json_dataset_path")
    sidecar_path = Path(sidecar_path_str) if sidecar_path_str else None

    metadata = read_json_if_exists(sidecar_path)

    source_id = str(
        raster_entry.get("source_id")
        or source_result.get("id")
        or _metadata_first(metadata, ["source_id", "source"])
        or "unknown"
    )

    provider = str(
        _metadata_first(metadata, ["provider", "source_provider"])
        or source_id.split("_")[0]
        or "unknown"
    )

    product = str(
        _metadata_first(metadata, ["product", "source_product"])
        or source_id.replace(f"{provider}_", "", 1)
    )

    variable = (
        _metadata_first(metadata, ["variable", "variable_name", "worldclim_variable_code"])
        or _infer_variable_from_name(dataset_path.stem)
    )

    resolution_m = (
        _as_int(_metadata_first(metadata, ["target_resolution_m", "resolution_m"]))
        or _infer_resolution_from_name(dataset_path.stem)
    )

    valid_range = _valid_range_from_metadata(metadata)
    months = _months_from_metadata(metadata)

    return LayerSpec(
        name=str(raster_entry.get("name") or dataset_path.stem),
        path=dataset_path,
        provider=provider,
        product=product,
        source_id=source_id,
        variable=str(variable) if variable is not None else None,
        variable_description=_metadata_first(
            metadata,
            ["variable_description", "description"],
        ),
        unit=_metadata_first(metadata, ["unit", "variable_unit"]),
        valid_range=valid_range,
        aoi=(
            _metadata_first(
                metadata,
                [
                    "output_aoi_name",
                    "output_aoi",
                    "aoi_name",
                    "domain_name",
                    "domain",
                    "aoi",
                ],
            )
            or manifest.get("run_aoi_name")
        ),
        resolution_m=resolution_m or _as_int(manifest.get("run_resolution_m")),
        crs=_metadata_first(metadata, ["crs", "target_crs"]),
        nodata=_as_float(_metadata_first(metadata, ["nodata"])),
        dtype=_metadata_first(metadata, ["dtype"]),
        aggregation_name=_metadata_first(
            metadata,
            ["aggregation_name", "temporal_aggregation_name"],
        ),
        aggregation_metric=_metadata_first(
            metadata,
            ["aggregation_metric", "metric"],
        ),
        months=months,
        year=_as_int(_metadata_first(metadata, ["year"])),
        period=_metadata_first(metadata, ["future_period", "period"]),
        gcm=_metadata_first(metadata, ["gcm"]),
        ssp=_metadata_first(metadata, ["ssp"]),
        layer_type=_metadata_first(
            metadata,
            ["dataset_layer_structure", "layer_structure", "layer_type"],
        ),
        source_config_path=source_result.get("config"),
        sidecar_metadata_path=sidecar_path,
        original_path=original_path,
        dataset_path=dataset_path,
        metadata=metadata,
    )


def build_layer_catalog_from_manifest(
    manifest: dict[str, Any],
) -> list[LayerSpec]:
    """
    Build a list of LayerSpec objects from a dataset manifest.
    """
    sources_by_id = {
        source["id"]: source
        for source in manifest.get("sources", [])
        if "id" in source
    }

    layers: list[LayerSpec] = []

    for raster_entry in manifest.get("rasters", []):
        source_id = raster_entry.get("source_id")
        source_result = sources_by_id.get(source_id, {})

        layers.append(
            build_layer_spec_from_raster_entry(
                raster_entry=raster_entry,
                source_result=source_result,
                manifest=manifest,
            )
        )

    return layers


def summarize_layer_catalog(layers: list[LayerSpec]) -> dict[str, Any]:
    """
    Small summary useful for logs and metadata.
    """
    providers = sorted({layer.provider for layer in layers})
    products = sorted({layer.product for layer in layers})
    variables = sorted({layer.variable for layer in layers if layer.variable is not None})
    aois = sorted({layer.aoi for layer in layers if layer.aoi is not None})
    resolutions = sorted({layer.resolution_m for layer in layers if layer.resolution_m is not None})

    return {
        "n_layers": len(layers),
        "providers": providers,
        "products": products,
        "variables": variables,
        "aois": aois,
        "resolutions_m": resolutions,
    }