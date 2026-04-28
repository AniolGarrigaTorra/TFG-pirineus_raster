from rasterio.enums import Resampling


_RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "min": Resampling.min,
    "max": Resampling.max,
    "sum": Resampling.sum,
}


def get_resampling_method(method_name: str) -> Resampling:
    """
    Convert a YAML resampling method string into a rasterio Resampling enum.
    """
    method_name = method_name.lower().strip()

    if method_name not in _RESAMPLING_METHODS:
        raise ValueError(
            f"Unsupported resampling method: {method_name}. "
            f"Supported methods: {sorted(_RESAMPLING_METHODS)}"
        )

    return _RESAMPLING_METHODS[method_name]


def get_variable_resampling_method(source_cfg: dict, variable: str) -> Resampling:
    """
    Read the resampling method for a variable from the source YAML.
    """
    resampling_cfg = source_cfg.get("resampling", {})

    default_method = resampling_cfg.get("default", "nearest")
    by_variable = resampling_cfg.get("by_variable", {})

    method_name = by_variable.get(variable, default_method)

    return get_resampling_method(method_name)


def get_variable_resampling_method_name(source_cfg: dict, variable: str) -> str:
    """
    Return the string name of the resampling method for metadata.
    """
    resampling_cfg = source_cfg.get("resampling", {})

    default_method = resampling_cfg.get("default", "nearest")
    by_variable = resampling_cfg.get("by_variable", {})

    return by_variable.get(variable, default_method)