from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
                    ],
                },
            )
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = _read_json_body(self)
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
    print("Endpoints: /api/catalog, /api/validate-run, /api/render-run")
    print("==============================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pirineus Raster Config API.")
    finally:
        server.server_close()

