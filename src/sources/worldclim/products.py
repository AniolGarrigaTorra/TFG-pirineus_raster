from dataclasses import dataclass


@dataclass(frozen=True)
class WorldClimProduct:
    product: str
    layer_structure: str
    zip_variable_mode: str
    supports_temporal_aggregation: bool


WORLDCLIM_PRODUCTS = {
    "v2_1_climate_normals": WorldClimProduct(
        product="v2_1_climate_normals",
        layer_structure="monthly_climatology",
        zip_variable_mode="per_variable",
        supports_temporal_aggregation=True,
    ),
    "v2_1_bioclim": WorldClimProduct(
        product="v2_1_bioclim",
        layer_structure="static_index_set",
        zip_variable_mode="single_zip",
        supports_temporal_aggregation=False,
    ),
    "v2_1_elevation": WorldClimProduct(
        product="v2_1_elevation",
        layer_structure="static_single",
        zip_variable_mode="single_zip",
        supports_temporal_aggregation=False,
    ),
    "cruts4_09_monthly": WorldClimProduct(
        product="cruts4_09_monthly",
        layer_structure="monthly_time_series",
        zip_variable_mode="variable_period",
        supports_temporal_aggregation=True,
    ),
    "cmip6_future": WorldClimProduct(
        product="cmip6_future",
        layer_structure="future_monthly_multiband",
        zip_variable_mode="direct_file",
        supports_temporal_aggregation=True,
    ),
}


def get_worldclim_product(product: str) -> WorldClimProduct:
    try:
        return WORLDCLIM_PRODUCTS[product]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported WorldClim product: {product}. "
            f"Supported products: {sorted(WORLDCLIM_PRODUCTS)}"
        ) from exc


def get_layer_structure(source_cfg: dict) -> str:
    dataset_cfg = source_cfg.get("dataset", {})
    product = source_cfg["source"]["product"]

    if "layer_structure" in dataset_cfg:
        return dataset_cfg["layer_structure"]

    return get_worldclim_product(product).layer_structure