from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_run_overrides_to_source_cfg(
    source_cfg: dict[str, Any],
    run_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply run-level defaults/overrides to a source config.

    Purpose:
      - Source configs keep sensible defaults so they can run standalone.
      - Run configs can override global execution parameters such as:
          * final output AOI
          * clipping AOI
          * final target resolution

    This keeps source configs focused on source-specific details.
    """
    cfg = deepcopy(source_cfg)

    if run_cfg is None:
        return cfg

    run = run_cfg.get("run", {}) or {}

    domains = cfg.setdefault("domains", {})
    processing = cfg.setdefault("processing", {})

    # ------------------------------------------------------------
    # Output AOI override
    # ------------------------------------------------------------
    # Preferred run-level field:
    #   run.aoi_config
    #
    # This is the final AOI/grid AOI for the generated features.
    if run.get("aoi_config"):
        domains["output_aoi_config"] = run["aoi_config"]

    # ------------------------------------------------------------
    # Optional clip AOI override
    # ------------------------------------------------------------
    # Usually we keep clip_aoi_config from the source config because
    # sources may need a larger/raw clipping domain.
    #
    # But if the run explicitly defines run.clip_aoi_config, we apply it.
    if run.get("clip_aoi_config"):
        domains["clip_aoi_config"] = run["clip_aoi_config"]

    # ------------------------------------------------------------
    # Target resolution override
    # ------------------------------------------------------------
    # Preferred run-level field:
    #   run.resolution_m
    if run.get("resolution_m") is not None:
        processing["target_resolution_m"] = int(run["resolution_m"])

    return cfg