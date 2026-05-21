from __future__ import annotations

from rasterio.enums import Resampling


RASTER_RESAMPLING_METHODS: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
}

CONSERVATIVE_RESAMPLING_METHODS = {
    "conservative_sum": Resampling.average,
    "extensive_sum": Resampling.average,
}

EXECUTABLE_RESAMPLING_METHODS = {
    **RASTER_RESAMPLING_METHODS,
    **CONSERVATIVE_RESAMPLING_METHODS,
}

ADVANCED_INTERPOLATION_METHODS = [
    {
        "name": "ordinary_kriging",
        "label": "Ordinary kriging",
        "status": "not_implemented",
        "notes": "Requires a geostatistical interpolation backend and variogram model.",
    },
    {
        "name": "universal_kriging",
        "label": "Universal kriging",
        "status": "not_implemented",
        "notes": "Requires drift terms/covariates and a geostatistical backend.",
    },
    {
        "name": "regression_kriging",
        "label": "Regression kriging",
        "status": "not_implemented",
        "notes": "Requires predictor rasters and residual kriging.",
    },
    {
        "name": "co_kriging",
        "label": "Co-kriging",
        "status": "not_implemented",
        "notes": "Requires secondary correlated variables.",
    },
    {
        "name": "idw",
        "label": "Inverse distance weighting",
        "status": "not_implemented",
        "notes": "Point interpolation method, not a direct raster warp method.",
    },
    {
        "name": "thin_plate_spline",
        "label": "Thin plate spline",
        "status": "not_implemented",
        "notes": "Point/surface interpolation method, not a direct raster warp method.",
    },
]

VALUE_SEMANTICS = [
    "categorical",
    "ordinal",
    "intensive",
    "intensive_depth",
    "percentage",
    "fraction",
    "extensive",
    "count",
]


def executable_resampling_names() -> list[str]:
    return sorted(EXECUTABLE_RESAMPLING_METHODS)


def get_resampling_enum(method_name: str | None) -> Resampling:
    if method_name is None:
        method_name = "nearest"

    method_name = str(method_name).lower()
    if method_name not in EXECUTABLE_RESAMPLING_METHODS:
        valid = ", ".join(executable_resampling_names())
        raise ValueError(
            f"Unsupported resampling method '{method_name}'. "
            f"Valid executable methods are: {valid}"
        )

    return EXECUTABLE_RESAMPLING_METHODS[method_name]


def is_conservative_resampling(method_name: str | None) -> bool:
    return str(method_name or "").lower() in CONSERVATIVE_RESAMPLING_METHODS
