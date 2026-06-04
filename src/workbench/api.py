from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pyproj import Transformer

from src.io.config import get_repo_root, load_yaml, resolve_path
from src.make_grid import create_grid
from src.pipeline.project_overrides import apply_run_overrides_to_project_cfg, normalize_crs
from src.workbench.catalog import workbench_catalog
from src.workbench.compiler import (
    render_run_config_yaml,
    validate_researcher_run_config,
)


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw)


def _extract_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    run_cfg = payload.get("run_config", payload)
    if not isinstance(run_cfg, dict):
        raise ValueError("Request body must be a run config object.")
    return run_cfg


def _sanitize_config_name(value: Any) -> str:
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        raise ValueError("Config name cannot be empty.")
    return name


def _bounds_epsg4326(crs: str, bounds: dict[str, float]) -> dict[str, float]:
    if normalize_crs(crs) == "EPSG:4326":
        return {
            "xmin": bounds["xmin"],
            "xmax": bounds["xmax"],
            "ymin": bounds["ymin"],
            "ymax": bounds["ymax"],
        }

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = transformer.transform_bounds(
        bounds["xmin"],
        bounds["ymin"],
        bounds["xmax"],
        bounds["ymax"],
        densify_pts=21,
    )
    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
    }


def _save_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    run_cfg = _extract_run_config(payload)
    report = validate_researcher_run_config(run_cfg)
    yaml_text = render_run_config_yaml(run_cfg)

    run_section = run_cfg.get("run", {}) if isinstance(run_cfg.get("run"), dict) else {}
    name = _sanitize_config_name(payload.get("name") or run_section.get("name"))

    repo_root = get_repo_root()
    run_path = repo_root / "configs" / "runs" / f"{name}.yaml"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("w", encoding="utf-8") as f:
        f.write(yaml_text)

    return {
        "ok": report["ok"],
        "validation": report,
        "path": str(run_path.relative_to(repo_root)),
        "yaml": yaml_text,
    }


def _create_aoi_config(payload: dict[str, Any]) -> dict[str, Any]:
    name = _sanitize_config_name(payload.get("name"))
    crs = normalize_crs(payload.get("crs"))
    bounds_payload = payload.get("bounds", {}) or {}
    bounds = {
        key: float(bounds_payload[key])
        for key in ["xmin", "xmax", "ymin", "ymax"]
    }

    if bounds["xmin"] >= bounds["xmax"]:
        raise ValueError("AOI bounds must satisfy xmin < xmax.")
    if bounds["ymin"] >= bounds["ymax"]:
        raise ValueError("AOI bounds must satisfy ymin < ymax.")

    repo_root = get_repo_root()
    aoi_path = repo_root / "configs" / "aoi" / f"{name}.yaml"
    if aoi_path.exists() and not bool(payload.get("overwrite", False)):
        raise FileExistsError(
            f"AOI config already exists: {aoi_path}. Choose another name."
        )

    cfg = {
        "name": name,
        "description": payload.get("description") or "Workbench-created AOI.",
        "crs": crs,
        "bounds": bounds,
        "bounds_epsg4326": _bounds_epsg4326(crs, bounds),
    }

    aoi_path.parent.mkdir(parents=True, exist_ok=True)
    with aoi_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    return {
        "ok": True,
        "aoi": {
            "name": name,
            "path": str(aoi_path.relative_to(repo_root)),
            "description": cfg["description"],
            "crs": crs,
            "bounds": bounds,
        },
    }


def _create_grid(payload: dict[str, Any]) -> dict[str, Any]:
    project_path = resolve_path(
        payload.get("project_config", "configs/project.yaml"),
        must_exist=True,
    )
    aoi_path = resolve_path(payload["aoi_config"], must_exist=True)
    resolution_m = int(payload["resolution_m"])

    project_cfg = load_yaml(project_path)
    project_cfg["_config_path"] = str(project_path)
    project_cfg = apply_run_overrides_to_project_cfg(
        project_cfg,
        {"run": {"crs": payload.get("crs")}},
    )
    aoi_cfg = load_yaml(aoi_path)
    output_path = create_grid(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution=resolution_m,
        overwrite=bool(payload.get("overwrite", False)),
    )

    repo_root = get_repo_root()
    return {
        "ok": True,
        "grid_path": str(output_path.resolve().relative_to(repo_root)),
    }


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    project_config_path = "configs/project.yaml"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[workbench-api] {self.address_string()} - {format % args}")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                _json_response(self, 200, {"ok": True})
                return

            if path == "/api/catalog":
                _json_response(
                    self,
                    200,
                    workbench_catalog(self.project_config_path),
                )
                return

            _json_response(
                self,
                404,
                {
                    "ok": False,
                    "error": "Unknown endpoint.",
                    "endpoints": [
                        "GET /api/health",
                        "GET /api/catalog",
                        "POST /api/validate-run",
                        "POST /api/render-run",
                        "POST /api/save-run",
                        "POST /api/aoi-config",
                        "POST /api/grid",
                    ],
                },
            )
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = _read_json_body(self)

            if path == "/api/aoi-config":
                _json_response(self, 200, _create_aoi_config(payload))
                return

            if path == "/api/grid":
                _json_response(self, 200, _create_grid(payload))
                return

            run_cfg = _extract_run_config(payload)

            if path == "/api/validate-run":
                report = validate_researcher_run_config(run_cfg)
                _json_response(self, 200, report)
                return

            if path == "/api/render-run":
                report = validate_researcher_run_config(run_cfg)
                yaml_text = render_run_config_yaml(run_cfg)
                _json_response(
                    self,
                    200,
                    {
                        "ok": report["ok"],
                        "validation": report,
                        "yaml": yaml_text,
                    },
                )
                return

            if path == "/api/save-run":
                _json_response(self, 200, _save_run_config(payload))
                return

            _json_response(self, 404, {"ok": False, "error": "Unknown endpoint."})
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})


def serve_workbench_api(
    host: str = "127.0.0.1",
    port: int = 8765,
    project_config_path: str | Path = "configs/project.yaml",
) -> None:
    WorkbenchRequestHandler.project_config_path = str(project_config_path)
    server = ThreadingHTTPServer((host, int(port)), WorkbenchRequestHandler)

    print("==============================")
    print("Pirineus Raster Config API")
    print(f"URL: http://{host}:{port}")
    print(f"Project config: {project_config_path}")
    print(
        "Endpoints: /api/catalog, /api/validate-run, /api/render-run, "
        "/api/save-run, /api/aoi-config, /api/grid"
    )
    print("==============================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pirineus Raster Config API.")
    finally:
        server.server_close()
