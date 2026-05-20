from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _find_free_port(host: str, preferred_port: int) -> int:
    port = int(preferred_port)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                port += 1
                continue
        return port


def serve_ui(
    host: str = "127.0.0.1",
    api_port: int = 8765,
    ui_port: int = 5173,
    project_config_path: str = "configs/project.yaml",
    ui_dir: str | Path = "ui",
) -> None:
    ui_path = Path(ui_dir)
    if not ui_path.exists():
        raise FileNotFoundError(f"UI directory not found: {ui_path}")

    if shutil.which("npm") is None:
        raise RuntimeError("npm was not found in PATH. Install Node.js/npm before using serve-ui.")

    actual_api_port = _find_free_port(host, api_port)
    actual_ui_port = _find_free_port(host, ui_port)

    api_url = f"http://{host}:{actual_api_port}"
    ui_url = f"http://{host}:{actual_ui_port}"

    if actual_api_port != api_port:
        print(f"API port {api_port} is busy; using {actual_api_port}.")
    if actual_ui_port != ui_port:
        print(f"UI port {ui_port} is busy; using {actual_ui_port}.")

    api_cmd = [
        sys.executable,
        "-m",
        "src.cli.main",
        "serve-config-api",
        "--host",
        host,
        "--port",
        str(actual_api_port),
        "--project-config",
        project_config_path,
    ]

    env = os.environ.copy()
    env["VITE_CONFIG_API_TARGET"] = api_url

    ui_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(actual_ui_port),
    ]

    api_process = subprocess.Popen(api_cmd)
    ui_process: subprocess.Popen | None = None

    try:
        time.sleep(0.5)
        if api_process.poll() is not None:
            raise RuntimeError("Config API exited before the UI could start.")

        ui_process = subprocess.Popen(ui_cmd, cwd=ui_path, env=env)

        print("==============================")
        print("Pirineus Raster Workbench")
        print(f"Config API: {api_url}")
        print(f"UI:         {ui_url}")
        print("Press Ctrl+C to stop both servers.")
        print("==============================")

        ui_process.wait()
    except KeyboardInterrupt:
        print("\nStopping Pirineus Raster Workbench.")
    finally:
        for process in [ui_process, api_process]:
            if process is None or process.poll() is not None:
                continue
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
