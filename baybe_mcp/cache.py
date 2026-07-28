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
RESOURCE_FORMAT_VERSION = 2

DEFAULT_CACHE_DIR = ".baybe_mcp_cache"
DEFAULT_RECIPES_DIR = "recipes"
MANIFEST_NAME = "manifest.json"


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve the cache directory, defaulting to a relative dir in the cwd."""
    if cache_dir is None:
        return Path(DEFAULT_CACHE_DIR)
    return Path(cache_dir)


def resolve_recipes_dir(recipes_dir: str | Path | None = None) -> Path:
    """Resolve the user recipes directory, defaulting to a relative dir."""
    if recipes_dir is None:
        return Path(DEFAULT_RECIPES_DIR)
    return Path(recipes_dir)


def recipes_hash(recipes_dir: str | Path | None = None) -> str:
    """Return a stable hash of the user recipes directory.

    Hashes the relative paths and contents of all ``.md`` files so the cache is
    rebuilt whenever a user recipe is added, removed, or edited. An absent or
    empty directory hashes to a constant sentinel.
    """
    import hashlib

    directory = resolve_recipes_dir(recipes_dir)
    hasher = hashlib.sha256()
    if directory.is_dir():
        for path in sorted(directory.rglob("*.md")):
            if not path.is_file():
                continue
            rel = path.relative_to(directory).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(path.read_bytes())
            hasher.update(b"\0")
    return hasher.hexdigest()


def manifest_path(cache_dir: Path) -> Path:
    """Return the path to the manifest file within the cache directory."""
    return cache_dir / MANIFEST_NAME


def write_manifest(
    cache_dir: Path, baybe_version: str | None, recipes_digest: str | None = None
) -> dict:
    """Write the cache manifest and return its contents."""
    import time

    manifest = {
        "baybe_version": baybe_version,
        "resource_format_version": RESOURCE_FORMAT_VERSION,
        "recipes_hash": recipes_digest,
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


def is_cache_valid(
    cache_dir: Path,
    baybe_version: str | None = None,
    recipes_dir: str | Path | None = None,
) -> bool:
    """Return True if the cache matches the installed version, format, recipes.

    The cache is valid only when the manifest's BayBE version, resource format
    version, and user-recipes hash all match the current environment. The
    recipes hash guard ensures added/edited/removed user recipes trigger a
    rebuild.
    """
    if baybe_version is None:
        baybe_version = get_baybe_version()
    manifest = read_manifest(cache_dir)
    if manifest is None:
        return False
    return (
        manifest.get("baybe_version") == baybe_version
        and manifest.get("resource_format_version") == RESOURCE_FORMAT_VERSION
        and manifest.get("recipes_hash") == recipes_hash(recipes_dir)
    )
