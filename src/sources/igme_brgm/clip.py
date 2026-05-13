from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.validation import make_valid

gpd.options.io_engine = "fiona"

from src.io.paths import ensure_dir, get_source_interim_dir, get_source_raw_dir
from src.sources.igme_brgm.naming import (
    build_igme_brgm_clipped_vector_name,
    safe_name,
)


def _iter_enabled_datasets(source_cfg: dict):
    datasets = source_cfg.get("datasets", {})

    for dataset_name, dataset_cfg in datasets.items():
        if dataset_cfg.get("enabled", True):
            yield dataset_name, dataset_cfg


def _iter_enabled_layers(dataset_cfg: dict):
    layers = dataset_cfg.get("layers", {})

    for layer_name, layer_cfg in layers.items():
        if layer_cfg.get("enabled", True):
            yield layer_name, layer_cfg


def _get_source_resolution(source_cfg: dict) -> str:
    return str(source_cfg["processing"]["source_resolution"])


def _get_raw_extract_dir(
    project_cfg: dict,
    source_cfg: dict,
    dataset_name: str,
) -> Path:
    source = source_cfg["source"]

    raw_dir = get_source_raw_dir(
        project_cfg=project_cfg,
        provider=source["provider"],
        product=source["product"],
        source_resolution=_get_source_resolution(source_cfg),
    )

    return raw_dir / "extracted" / safe_name(dataset_name)


def _find_shapefile(
    extract_dir: Path,
    shapefile_name: str,
) -> Path:
    matches = list(extract_dir.rglob(shapefile_name))

    if not matches:
        available = [p.name for p in extract_dir.rglob("*.shp")]
        raise FileNotFoundError(
            f"Shapefile '{shapefile_name}' not found inside {extract_dir}.\n"
            f"Available shapefiles: {available}"
        )

    if len(matches) > 1:
        print(
            f"[clip] Warning: multiple matches for {shapefile_name}. "
            f"Using first: {matches[0]}"
        )

    return matches[0]


def _build_aoi_gdf(aoi_cfg: dict) -> gpd.GeoDataFrame:
    bounds = aoi_cfg["bounds"]

    geom = box(
        float(bounds["xmin"]),
        float(bounds["ymin"]),
        float(bounds["xmax"]),
        float(bounds["ymax"]),
    )

    return gpd.GeoDataFrame(
        {"name": [aoi_cfg["name"]]},
        geometry=[geom],
        crs=aoi_cfg["crs"],
    )


def _make_valid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    if gdf.empty:
        return gdf

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )

    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    return gdf


def _read_source_layer(
    shapefile_path: Path,
    source_crs: str | None,
) -> gpd.GeoDataFrame:
    print(f"[clip] Reading: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path, engine="fiona")

    if gdf.empty:
        print(f"[clip] Warning: empty source layer: {shapefile_path}")
        return gdf

    if gdf.crs is None:
        if source_crs is None:
            raise ValueError(
                f"Layer has no CRS and no source_crs was configured: {shapefile_path}"
            )

        print(f"[clip] Source layer has no CRS. Setting CRS to {source_crs}")
        gdf = gdf.set_crs(source_crs)

    return _make_valid_geometries(gdf)


def _get_clipped_output_path(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_name: str,
    dataset_name: str,
    layer_name: str,
) -> Path:
    source = source_cfg["source"]

    base_dir = (
        get_source_interim_dir(
            project_cfg=project_cfg,
            provider=source["provider"],
            product=source["product"],
        )
        / "clipped"
        / clip_aoi_name
        / _get_source_resolution(source_cfg)
        / safe_name(dataset_name)
    )

    ensure_dir(base_dir)

    return base_dir / build_igme_brgm_clipped_vector_name(
        dataset_name=dataset_name,
        layer_name=layer_name,
        domain_name=clip_aoi_name,
    )


def clip_igme_brgm_raw_files(
    project_cfg: dict,
    source_cfg: dict,
    clip_aoi_cfg: dict,
) -> list[Path]:
    """
    Clip configured IGME/BRGM vector layers to the clipping AOI.

    Output format is GeoPackage to avoid shapefile field-name and encoding
    limitations in intermediate files.
    """
    clip_cfg = source_cfg.get("clip", {})
    overwrite = bool(clip_cfg.get("overwrite_existing", False))

    clip_aoi_name = clip_aoi_cfg["name"]
    aoi_gdf = _build_aoi_gdf(clip_aoi_cfg)

    print(f"[clip] AOI: {clip_aoi_name}")
    print(f"[clip] AOI CRS: {aoi_gdf.crs}")

    output_paths: list[Path] = []

    for dataset_name, dataset_cfg in _iter_enabled_datasets(source_cfg):
        extract_dir = _get_raw_extract_dir(
            project_cfg=project_cfg,
            source_cfg=source_cfg,
            dataset_name=dataset_name,
        )

        if not extract_dir.exists():
            raise FileNotFoundError(
                f"Extracted directory not found: {extract_dir}\n"
                "Run the download stage first."
            )

        dataset_source_crs = dataset_cfg.get(
            "source_crs",
            source_cfg["source"].get("source_crs"),
        )

        for layer_name, layer_cfg in _iter_enabled_layers(dataset_cfg):
            shapefile_name = layer_cfg["shapefile"]

            output_path = _get_clipped_output_path(
                project_cfg=project_cfg,
                source_cfg=source_cfg,
                clip_aoi_name=clip_aoi_name,
                dataset_name=dataset_name,
                layer_name=layer_name,
            )

            if output_path.exists() and not overwrite:
                print(f"[clip] Exists, skipping: {output_path}")
                output_paths.append(output_path)
                continue

            shapefile_path = _find_shapefile(
                extract_dir=extract_dir,
                shapefile_name=shapefile_name,
            )

            gdf = _read_source_layer(
                shapefile_path=shapefile_path,
                source_crs=dataset_source_crs,
            )

            if gdf.empty:
                clipped = gdf
            else:
                gdf = gdf.to_crs(aoi_gdf.crs)
                clipped = gpd.clip(gdf, aoi_gdf)
                clipped = _make_valid_geometries(clipped)

            ensure_dir(output_path.parent)

            gpkg_layer_name = safe_name(layer_name)

            print(f"[clip] Writing clipped layer:")
            print(f"  Dataset: {dataset_name}")
            print(f"  Layer:   {layer_name}")
            print(f"  Count:   {len(clipped)}")
            print(f"  Out:     {output_path}")

            clipped.to_file(
                output_path,
                layer=gpkg_layer_name,
                driver="GPKG",
                engine="fiona",
            )

            output_paths.append(output_path)

    return output_paths