"""Resource cache: manifest, staleness detection, and cache-dir resolution.

The cache stores all built resources plus a manifest. Cached resources are only
considered valid when the manifest agrees with the installed BayBE version and
the current resource format version. This makes caches portable: they can be
built on one machine and copied to another (e.g. one without internet access).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from baybe_mcp.version import get_baybe_version

logger = logging.getLogger(__name__)

# Bump whenever the structure of any built resource changes, so that caches
# produced by older code are rebuilt even if the BayBE version is unchanged.
RESOURCE_FORMAT_VERSION = 1

DEFAULT_CACHE_DIR = ".baybe_mcp_cache"
MANIFEST_NAME = "manifest.json"


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the cache directory, defaulting to a relative dir in the cwd."""
    if cache_dir is None:
        return Path(DEFAULT_CACHE_DIR)
    return Path(cache_dir)


def manifest_path(cache_dir: Path) -> Path:
    """Return the path to the manifest file within the cache directory."""
    return cache_dir / MANIFEST_NAME


def write_manifest(cache_dir: Path, baybe_version: str | None) -> dict:
    """Write the cache manifest and return its contents."""
    import time

    manifest = {
        "baybe_version": baybe_version,
        "resource_format_version": RESOURCE_FORMAT_VERSION,
        "built_at": time.time(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(cache_dir: Path) -> dict | None:
    """Read the cache manifest, or None if missing/corrupt."""
    path = manifest_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_cache_valid(cache_dir: Path, baybe_version: str | None = None) -> bool:
    """Return True if the cache matches the installed version and format.

    The cache is valid only when the manifest's BayBE version and resource
    format version both match the current environment.
    """
    if baybe_version is None:
        baybe_version = get_baybe_version()
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return False
    return (
        manifest.get("baybe_version") == baybe_version
        and manifest.get("resource_format_version") == RESOURCE_FORMAT_VERSION
    )
