"""Resource build orchestration.

Each resource (types, schema, docs, concepts, recipes) registers a builder that
writes its artifacts into the cache directory. ``build_resources`` runs all
registered builders and writes the manifest. The same routine is used both by
the standalone ``build`` command and by the server at startup, so there is a
single build code path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from baybe_mcp.cache import (
    recipes_hash,
    resolve_cache_dir,
    resolve_recipes_dir,
    write_manifest,
)
from baybe_mcp.version import get_baybe_version

logger = logging.getLogger(__name__)

# A builder takes the cache directory and the resolved user recipes directory
# and writes its artifacts into the cache.
Builder = Callable[[Path, Path], None]

_BUILDERS: list[tuple[str, Builder]] = []


def register_builder(name: str, builder: Builder) -> None:
    """Register a resource builder under a name (idempotent by name)."""
    for i, (existing, _) in enumerate(_BUILDERS):
        if existing == name:
            _BUILDERS[i] = (name, builder)
            return
    _BUILDERS.append((name, builder))


def build_resources(
    cache_dir: str | Path | None = None, recipes_dir: str | Path | None = None
) -> Path:
    """Build all registered resources into the cache directory.

    Writes the manifest last so a partially built cache is not considered valid.
    The manifest records a hash of the user recipes directory so the cache is
    rebuilt when user recipes change. Returns the resolved cache directory.
    """
    cache_dir = resolve_cache_dir(cache_dir)
    recipes_directory = resolve_recipes_dir(recipes_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building resources into %s", cache_dir)

    for name, builder in _BUILDERS:
        logger.info("Building resource: %s", name)
        builder(cache_dir, recipes_directory)

    write_manifest(cache_dir, get_baybe_version(), recipes_hash(recipes_dir))
    logger.info("Resource build complete.")
    return cache_dir
