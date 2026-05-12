from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.io.paths import get_source_clipped_dir, get_feature_output_dir
from src.pipeline.raster_ops import (
    load_grid_context,
    print_grid_context,
    read_raster_to_grid,
    write_feature_raster,
    build_static_feature_metadata,
    get_variable_resampling_method,
    get_variable_resampling_method_name,
)
from src.sources.copernicus.naming import (
    validate_copernicus_source_config,
    get_enabled_variable_items,
    get_source_resolution,
    build_copernicus_clipped_name,
    build_copernicus_feature_name,
)


def _get_target_resolution_m(source_cfg: dict) -> int:
    return int(source_cfg["processing"]["target_resolution_m"])


def _get_output_options(project_cfg: dict, source_cfg: dict) -> dict[str, Any]:
    output_cfg = source_cfg.get("output", {}) or {}

    return {
        "output_dtype": str(output_cfg.get("dtype", "float32")),
        "nodata": float(project_cfg.get("nodata", -9999.0)),
        "compression": str(output_cfg.get("compression", "LZW")),
        "write_sidecar": bool(output_cfg.get("write_sidecar_json", True)),
    }


def _get_clipped_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    variable: str,
) -> Path:
    source = source_cfg["source"]
    source_resolution = get_source_resolution(source_cfg)

    clipped_dir = get_source_clipped_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=clip_aoi_name,
        source_resolution=source_resolution,
        variable=variable,
    )

    clipped_name = build_copernicus_clipped_name(
        source_cfg=source_cfg,
        variable=variable,
        domain_name=clip_aoi_name,
    )

    return clipped_dir / clipped_name


def _get_output_path(
    project_cfg: dict,
    source_cfg: dict,
    output_aoi_name: str,
    target_resolution_m: int,
    variable: str,
) -> Path:
    source = source_cfg["source"]

    output_dir = get_feature_output_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    output_name = build_copernicus_feature_name(
        source_cfg=source_cfg,
        variable=variable,
        domain_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
    )

    return output_dir / output_name


def _postprocess_array(
    array: np.ndarray,
    variable_cfg: dict,
) -> np.ndarray:
    """
    Apply lightweight generic post-processing.

    Current operations:
      - optional valid_range masking
      - optional round_values for categorical rasters
    """
    out = array.astype(np.float32, copy=True)

    valid_range = variable_cfg.get("valid_range")
    if valid_range is not None:
        min_value = float(valid_range[0])
        max_value = float(valid_range[1])

        out = np.where(
            (out >= min_value) & (out <= max_value),
            out,
            np.nan,
        )

    if bool(variable_cfg.get("round_values", False)):
        finite = np.isfinite(out)
        out[finite] = np.rint(out[finite])

    return out.astype(np.float32)


def _build_copernicus_static_metadata(
    source_cfg: dict,
    variable: str,
    variable_cfg: dict,
    clip_aoi_name: str,
    output_aoi_name: str,
    target_resolution_m: int,
    resampling_method_name: str,
) -> dict:
    metadata = build_static_feature_metadata(
        source_cfg=source_cfg,
        layer_name=variable,
        layer_cfg=variable_cfg,
        clip_aoi_name=clip_aoi_name,
        output_aoi_name=output_aoi_name,
        target_resolution_m=target_resolution_m,
        resampling_method_name=resampling_method_name,
    )

    dataset_cfg = source_cfg.get("dataset", {}) or {}

    metadata.update(
        {
            "data_type": variable_cfg.get("data_type") or dataset_cfg.get("data_type"),
            "native_resolution_m": (
                variable_cfg.get("native_resolution_m")
                or dataset_cfg.get("native_resolution_m")
            ),
            "temporal_type": (
                variable_cfg.get("temporal", {}).get("type")
                if isinstance(variable_cfg.get("temporal"), dict)
                else None
            ),
            "reference_year": (
                variable_cfg.get("temporal", {}).get("reference_year")
                if isinstance(variable_cfg.get("temporal"), dict)
                else source_cfg.get("source", {}).get("source_period")
            ),
        }
    )

    return {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


def build_copernicus_features(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
    output_aoi_cfg: dict,
) -> list[Path]:
    """
    Build final grid-aligned Copernicus static features.

    This supports both:
      - static_single: one enabled variable
      - static_multi: several enabled variables
    """
    validate_copernicus_source_config(source_cfg)

    clip_aoi_name = clip_aoi_cfg["name"]
    output_aoi_name = output_aoi_cfg["name"]
    target_resolution_m = _get_target_resolution_m(source_cfg)
    output_options = _get_output_options(project_cfg, source_cfg)

    grid = load_grid_context(
        project_cfg=project_cfg,
        aoi_cfg=output_aoi_cfg,
        resolution_m=target_resolution_m,
    )

    print("[build] Provider:", source_cfg["source"]["provider"])
    print("[build] Product:", source_cfg["source"]["product"])
    print("[build] Output AOI:", output_aoi_name)
    print("[build] Clip AOI:", clip_aoi_name)
    print("[build] Target resolution:", target_resolution_m)
    print_grid_context(grid, prefix="[build]")

    written_paths: list[Path] = []

    for variable, variable_cfg in get_enabled_variable_items(source_cfg):
        scale_factor = float(variable_cfg.get("scale_factor", 1.0))

        resampling = get_variable_resampling_method(
            source_cfg=source_cfg,
            variable=variable,
        )
        resampling_name = get_variable_resampling_method_name(
            source_cfg=source_cfg,
            variable=variable,
        )

        clipped_path = _get_clipped_path(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            clip_aoi_name=clip_aoi_name,
            variable=variable,
        )

        if not clipped_path.exists():
            if not bool(variable_cfg.get("required", True)):
                print(
                    f"[build] Optional clipped Copernicus raster missing for "
                    f"variable={variable}. Skipping: {clipped_path}"
                )
                continue

            raise FileNotFoundError(
                f"Missing clipped Copernicus raster: {clipped_path}\n"
                "Run the clip stage first."
            )

        output_path = _get_output_path(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            variable=variable,
        )

        print("==============================")
        print(f"[build] Variable: {variable}")
        print(f"[build] Description: {variable_cfg.get('description', '')}")
        print(f"[build] Data type: {variable_cfg.get('data_type')}")
        print(f"[build] Native resolution: {variable_cfg.get('native_resolution_m')}")
        print(f"[build] Scale factor: {scale_factor}")
        print(f"[build] Resampling: {resampling_name}")
        print(f"[build] Clipped path: {clipped_path}")
        print(f"[build] Output path: {output_path}")

        grid_array = read_raster_to_grid(
            raster_path=clipped_path,
            grid=grid,
            resampling=resampling,
            band=1,
            scale_factor=scale_factor,
        )

        grid_array = _postprocess_array(
            array=grid_array,
            variable_cfg=variable_cfg,
        )

        metadata = _build_copernicus_static_metadata(
            source_cfg=source_cfg,
            variable=variable,
            variable_cfg=variable_cfg,
            clip_aoi_name=clip_aoi_name,
            output_aoi_name=output_aoi_name,
            target_resolution_m=target_resolution_m,
            resampling_method_name=resampling_name,
        )

        written_path = write_feature_raster(
            output_path=output_path,
            array=grid_array,
            grid=grid,
            metadata=metadata,
            **output_options,
            validate=True,
        )

        print(f"[build] Written: {written_path}")
        written_paths.append(written_path)

    return written_paths