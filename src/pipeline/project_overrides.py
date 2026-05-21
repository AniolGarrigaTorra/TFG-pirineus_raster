from __future__ import annotations

from copy import deepcopy
from typing import Any

from rasterio.crs import CRS


def normalize_crs(value: Any) -> str:
    return CRS.from_user_input(value).to_string()


def apply_run_overrides_to_project_cfg(
    project_cfg: dict[str, Any],
    run_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(project_cfg)

    if run_cfg is None:
        return cfg

    run = run_cfg.get("run", {}) or {}
    crs = run.get("crs") or run.get("target_crs") or run.get("output_crs")
    if crs is None:
        return cfg

    normalized = normalize_crs(crs)
    if normalized != cfg.get("crs"):
        cfg["_default_crs"] = cfg.get("crs")
        cfg["_crs_overridden"] = True
        cfg["_grid_crs_suffix"] = normalized.lower().replace(":", "")

    cfg["crs"] = normalized
    return cfg
