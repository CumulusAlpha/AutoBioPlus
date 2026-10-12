"""Resolve the OpenPI client from the active or standalone environment."""

import sys
from pathlib import Path


def load_openpi_client():
    """Import the websocket client, using the bundled checkout as a fallback."""
    try:
        from openpi_client import websocket_client_policy
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        openpi_root = repo_root / "thirdparty" / "openpi"
        sys.path.append(str(openpi_root / "packages" / "openpi-client" / "src"))
        site_packages = sorted(
            (openpi_root / ".venv" / "lib").glob("python*/site-packages")
        )
        if site_packages:
            sys.path.append(str(site_packages[-1]))
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise SystemExit(
                "Missing openpi-client dependencies. Sync the standalone "
                "thirdparty/openpi environment first."
            ) from exc
    return websocket_client_policy
