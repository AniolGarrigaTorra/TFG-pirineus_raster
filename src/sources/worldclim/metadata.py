def build_worldclim_source_metadata(source_cfg: dict) -> dict:
    source = source_cfg["source"]
    dataset = source_cfg.get("dataset", {})

    return {
        "source_provider": source.get("provider", "worldclim"),
        "source_id": source.get("id", ""),
        "source_product": source.get("product", ""),
        "source_product_group": source.get("product_group", ""),
        "source_version": str(source.get("version", "")),
        "source_description": source.get("description", ""),
        "source_page_url": source.get("page_url", ""),
        "source_documentation_url": source.get("documentation_url", ""),
        "source_base_url": source.get("base_url", ""),
        "source_crs": source.get("source_crs", ""),
        "source_period": source.get("source_period", ""),
        "source_citation": source.get("citation", ""),
        "dataset_layer_structure": dataset.get("layer_structure", ""),
        "dataset_file_format": dataset.get("file_format", ""),
    }


