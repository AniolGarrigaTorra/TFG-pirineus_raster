from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.pipeline.feature_writer import write_feature_raster
from src.pipeline.grid_context import load_grid_context
from src.pipeline.layer_catalog import build_layer_catalog_from_manifest
from src.pipeline.layer_spec import LayerSpec
from src.pipeline.raster_math import (
    evaluate_raster_expression,
    read_raster_array_as_nan,
)


def _load_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def _months_match(layer_months: list[int] | None, requested_months: list[int] | None) -> bool:
    if requested_months is None:
        return True

    if layer_months is None:
        return False

    return list(layer_months) == list(requested_months)


def _matches_layer_query(layer: LayerSpec, query: dict[str, Any]) -> bool:
    """
    Return True if a LayerSpec matches an input query from run_config.
    """
    if "provider" in query and layer.provider != query["provider"]:
        return False

    if "product" in query and layer.product != query["product"]:
        return False

    if "source_id" in query and layer.source_id != query["source_id"]:
        return False

    if "variable" in query and layer.variable != query["variable"]:
        return False

    if "aggregation_name" in query and layer.aggregation_name != query["aggregation_name"]:
        return False

    if "aggregation_metric" in query and layer.aggregation_metric != query["aggregation_metric"]:
        return False

    if "months" in query and not _months_match(layer.months, query["months"]):
        return False

    if "gcm" in query and layer.gcm != query["gcm"]:
        return False

    if "ssp" in query and layer.ssp != query["ssp"]:
        return False

    if "period" in query and layer.period != query["period"]:
        return False

    return True


def _find_layer(
    layers: list[LayerSpec],
    query: dict[str, Any],
    input_name: str,
) -> LayerSpec:
    matches = [
        layer
        for layer in layers
        if _matches_layer_query(layer, query)
    ]

    if not matches:
        raise ValueError(
            f"No layer found for derived feature input '{input_name}' "
            f"with query: {query}"
        )

    if len(matches) > 1:
        match_names = [layer.name for layer in matches]
        raise ValueError(
            f"Ambiguous input '{input_name}'. Query matched {len(matches)} layers: "
            f"{match_names}. Add provider/product/source_id/months/period to disambiguate."
        )

    return matches[0]


def _build_derived_metadata(
    derived_cfg: dict[str, Any],
    input_layers: dict[str, LayerSpec],
    output_path: Path,
    run_name: str,
) -> dict[str, Any]:
    return {
        "provider": "derived",
        "product": "derived_features",
        "source_id": "derived",
        "layer_type": "derived",
        "variable": derived_cfg["name"],
        "variable_description": derived_cfg.get("description"),
        "unit": derived_cfg.get("unit"),
        "expression": derived_cfg["expression"],
        "run_name": run_name,
        "inputs": {
            input_name: {
                "layer_name": layer.name,
                "path": str(layer.path),
                "provider": layer.provider,
                "product": layer.product,
                "source_id": layer.source_id,
                "variable": layer.variable,
                "aggregation_name": layer.aggregation_name,
                "aggregation_metric": layer.aggregation_metric,
                "months": layer.months,
                "gcm": layer.gcm,
                "ssp": layer.ssp,
                "period": layer.period,
            }
            for input_name, layer in input_layers.items()
        },
        "output_path": str(output_path),
    }


def _append_derived_raster_to_manifest(
    manifest: dict[str, Any],
    raster_entry: dict[str, Any],
    layer_entry: dict[str, Any],
) -> None:
    manifest.setdefault("rasters", []).append(raster_entry)
    manifest.setdefault("layer_catalog", []).append(layer_entry)

    manifest["n_rasters"] = len(manifest.get("rasters", []))

    layer_summary = manifest.get("layer_summary", {})
    layer_summary["n_layers"] = len(manifest.get("layer_catalog", []))
    manifest["layer_summary"] = layer_summary


def build_derived_features(
    run_cfg: dict[str, Any],
    project_cfg: dict[str, Any],
    dataset_dir: Path,
    output_aoi_cfg: dict[str, Any],
) -> list[Path]:
    """
    Build derived raster features defined in run_cfg['derived_features'].

    Derived features are evaluated from already generated dataset rasters.
    """
    derived_features = run_cfg.get("derived_features", [])

    if not derived_features:
        return []

    manifest = _load_manifest(dataset_dir)
    layers = build_layer_catalog_from_manifest(manifest)

    rasters_dir = dataset_dir / "rasters"
    rasters_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_cfg["run"]["name"]
    target_resolution_m = int(run_cfg["run"]["resolution_m"])

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    nodata = float(project_cfg.get("nodata", -9999.0))

    written_paths: list[Path] = []

    for derived_cfg in derived_features:
        name = derived_cfg["name"]
        expression = derived_cfg["expression"]
        inputs_cfg = derived_cfg.get("inputs", {})

        if not inputs_cfg:
            raise ValueError(f"Derived feature '{name}' has no inputs.")

        print("==============================")
        print(f"[derived] Feature: {name}")
        print(f"[derived] Expression: {expression}")

        input_layers: dict[str, LayerSpec] = {}
        input_arrays = {}

        for input_name, query in inputs_cfg.items():
            layer = _find_layer(
                layers=layers,
                query=query,
                input_name=input_name,
            )

            input_layers[input_name] = layer

            array, _ = read_raster_array_as_nan(layer.path)
            input_arrays[input_name] = array

            print(f"[derived] Input {input_name}: {layer.name}")

        result = evaluate_raster_expression(
            expression=expression,
            variables=input_arrays,
        )

        output_name = f"derived_{name}.tif"
        output_path = rasters_dir / output_name

        metadata = _build_derived_metadata(
            derived_cfg=derived_cfg,
            input_layers=input_layers,
            output_path=output_path,
            run_name=run_name,
        )

        output_dtype = derived_cfg.get("output_dtype", "float32")

        written_path = write_feature_raster(
            output_path=output_path,
            array=result,
            grid=grid,
            metadata=metadata,
            output_dtype=output_dtype,
            nodata=nodata,
            compression="LZW",
            write_sidecar=True,
            validate=True,
        )

        sidecar_path = written_path.with_suffix(".json")

        raster_entry = {
            "name": written_path.stem,
            "source_id": "derived",
            "original_path": str(written_path),
            "dataset_path": str(written_path),
            "sidecar_json_original_path": str(sidecar_path),
            "sidecar_json_dataset_path": str(sidecar_path),
        }

        layer_entry = {
            "name": written_path.stem,
            "path": str(written_path),
            "provider": "derived",
            "product": "derived_features",
            "source_id": "derived",
            "variable": name,
            "variable_description": derived_cfg.get("description"),
            "unit": derived_cfg.get("unit"),
            "aoi": manifest.get("run_aoi_name"),
            "resolution_m": manifest.get("run_resolution_m"),
            "crs": str(grid.crs),
            "nodata": nodata,
            "dtype": output_dtype,
            "layer_type": "derived",
            "sidecar_metadata_path": str(sidecar_path),
            "original_path": str(written_path),
            "dataset_path": str(written_path),
            "metadata": metadata,
        }

        _append_derived_raster_to_manifest(
            manifest=manifest,
            raster_entry=raster_entry,
            layer_entry=layer_entry,
        )

        written_paths.append(written_path)
        print(f"[derived] Written: {written_path}")

    _write_manifest(dataset_dir, manifest)

    return written_paths