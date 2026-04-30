from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LayerSpec:
    """
    Generic description of one raster layer produced by the pipeline.

    This object is intentionally provider-agnostic.

    It can represent:
    - one static raster, such as elevation
    - one bioclimatic index
    - one temporal aggregation
    - one future climate projection aggregation
    - one derived feature in a later phase
    """

    name: str
    path: Path

    provider: str
    product: str
    source_id: str

    variable: str | None = None
    variable_description: str | None = None
    unit: str | None = None
    valid_range: tuple[float, float] | None = None

    aoi: str | None = None
    resolution_m: int | None = None
    crs: str | None = None
    nodata: float | int | None = None
    dtype: str | None = None

    aggregation_name: str | None = None
    aggregation_metric: str | None = None
    months: list[int] | None = None

    year: int | None = None
    period: str | None = None
    gcm: str | None = None
    ssp: str | None = None

    layer_type: str | None = None
    source_config_path: str | None = None
    sidecar_metadata_path: Path | None = None

    original_path: Path | None = None
    dataset_path: Path | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key in [
            "path",
            "sidecar_metadata_path",
            "original_path",
            "dataset_path",
        ]:
            value = data.get(key)
            if value is not None:
                data[key] = str(value)

        if data.get("valid_range") is not None:
            data["valid_range"] = list(data["valid_range"])

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayerSpec":
        copied = dict(data)

        for key in [
            "path",
            "sidecar_metadata_path",
            "original_path",
            "dataset_path",
        ]:
            if copied.get(key) is not None:
                copied[key] = Path(copied[key])

        if copied.get("valid_range") is not None:
            copied["valid_range"] = tuple(copied["valid_range"])

        return cls(**copied)


def layer_specs_to_dicts(layers: list[LayerSpec]) -> list[dict[str, Any]]:
    return [layer.to_dict() for layer in layers]


def layer_specs_from_dicts(items: list[dict[str, Any]]) -> list[LayerSpec]:
    return [LayerSpec.from_dict(item) for item in items]