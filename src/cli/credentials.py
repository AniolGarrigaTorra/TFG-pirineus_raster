from __future__ import annotations

import getpass
import importlib.util
import os
import stat
from pathlib import Path


HDARC_PATH = Path.home() / ".hdarc"
HDARC_LABEL = "~/.hdarc"
WEKEO_LOGIN_URL = "https://wekeo.copernicus.eu/"
HDA_PYTHON_DOCS_URL = "https://help.wekeo.eu/en/articles/6751608-how-to-use-the-hda-api-in-python"
HDA_CLIENT_DOCS_URL = "https://hda.readthedocs.io/en/latest/installation.html"


def _read_hdarc(path: Path = HDARC_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                values[key.strip().lower()] = value.strip()
    except PermissionError:
        values["_permission_error"] = str(path)

    return values


def _permission_text(path: Path = HDARC_PATH) -> str | None:
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    return oct(mode)


def _has_env_credentials() -> bool:
    return bool(os.environ.get("HDA_USER") and os.environ.get("HDA_PASSWORD"))


def _has_hdarc_credentials(path: Path = HDARC_PATH) -> bool:
    values = _read_hdarc(path)
    return bool(values.get("user") and values.get("password"))


def _hda_package_available() -> bool:
    return importlib.util.find_spec("hda") is not None


def write_hdarc_interactive(path: Path = HDARC_PATH, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Use --force to replace it."
        )

    username = input("WEkEO username/email: ").strip()
    password = getpass.getpass("WEkEO password: ")

    if not username or not password:
        raise ValueError("Username and password must both be provided.")

    path.write_text(f"user:{username}\npassword:{password}\n", encoding="utf-8")
    path.chmod(0o600)

    print(f"WEkEO credentials written to {path}")
    print(f"Recommended permissions applied: chmod 600 {path}")


def print_credentials_report(setup: bool = False, force: bool = False) -> None:
    if setup:
        write_hdarc_interactive(force=force)

    hda_ok = _hda_package_available()
    env_ok = _has_env_credentials()
    hdarc_ok = _has_hdarc_credentials()
    hdarc_exists = HDARC_PATH.exists()
    permission = _permission_text()
    values = _read_hdarc()
    hdarc_permission_error = "_permission_error" in values

    if hdarc_ok:
        credentials_status = f"OK via {HDARC_LABEL}"
    elif env_ok:
        credentials_status = "OK via HDA_USER/HDA_PASSWORD"
    elif hdarc_permission_error:
        credentials_status = f"UNREADABLE in {HDARC_LABEL}"
    elif hdarc_exists:
        credentials_status = f"INCOMPLETE in {HDARC_LABEL}"
    else:
        credentials_status = "MISSING"

    print(f"WEkEO credentials: {credentials_status}")
    print(f"HDA package: {'OK' if hda_ok else 'MISSING'}")

    if hdarc_exists:
        print(f"Current permissions: {permission} {HDARC_LABEL}")
    print(f"Recommended permissions: chmod 600 {HDARC_LABEL}")

    if hdarc_permission_error:
        print(f"Fix permissions with: chmod 600 {HDARC_LABEL}")

    if values.get("url"):
        print("Warning: .hdarc contains a url entry. Current WEkEO guidance says to remove old broker URLs from .hdarc files created before March 2024.")

    credentials_ok = hdarc_ok or env_ok

    if hda_ok and credentials_ok:
        return

    if not credentials_ok:
        print("")
        print("How to configure WEkEO/HDA credentials")
        print("1. Create or verify your WEkEO account at:")
        print(f"   {WEKEO_LOGIN_URL}")
        print("2. Create the default HDA credentials file:")
        print(f"   {HDARC_LABEL}")
        print("   with this format:")
        print("   user:<your-wekeo-username-or-email>")
        print("   password:<your-wekeo-password>")
        print(f"3. Lock down the file: chmod 600 {HDARC_LABEL}")
        print("4. Or let this CLI create it interactively:")
        print("   pirineus-raster check-credentials --setup")

    if not hda_ok:
        print("")
        print("Install HDA in the active environment:")
        print("   pip install hda -U")
        print("   or: conda install conda-forge::hda")

    print("")
    print("Official references:")
    print(f"   {HDA_PYTHON_DOCS_URL}")
    print(f"   {HDA_CLIENT_DOCS_URL}")
