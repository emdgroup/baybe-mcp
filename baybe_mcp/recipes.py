"""Recipes: complete worked scenarios sourced from BayBE's examples/ folder.

The index of topics/files is discovered via the GitHub contents API at the
version tag; individual recipe files are fetched as raw jupytext ``.py``
content on demand. User-provided recipes (local ``.md`` files) are merged in by
the server at build time.
"""

from __future__ import annotations

import json
import logging

from baybe_mcp.net import fetch_text
from baybe_mcp.version import get_baybe_version

logger = logging.getLogger(__name__)

_API = "https://api.github.com/repos/emdgroup/baybe/contents"
_RAW = "https://raw.githubusercontent.com/emdgroup/baybe"

# Skipped because it is meta guidance, not a recipe scenario.
_SKIP_FILES = {"AGENTS.md"}


def _contents(version: str, path: str) -> list | None:
    """List a repo directory via the GitHub contents API at the given ref."""
    text = fetch_text(f"{_API}/{path}?ref={version}")
    if text is None:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def build_recipes_index() -> dict:
    """Build the index of recipe topics and their scenario files.

    Returns a dict with the version and, per topic, the list of files with
    their resource URIs and source. Falls back to a link when the listing is
    unavailable.
    """
    version = get_baybe_version()
    if version is None:
        return {"baybe_version": None, "note": "Version undetectable.", "topics": {}}

    entries = _contents(version, "examples")
    if entries is None:
        return {
            "baybe_version": version,
            "link": f"https://github.com/emdgroup/baybe/tree/{version}/examples",
            "note": "Recipe index unavailable offline; see the linked folder.",
            "topics": {},
        }

    topics: dict = {}
    for entry in entries:
        if entry.get("type") != "dir":
            continue
        topic = entry["name"]
        files = _contents(version, f"examples/{topic}")
        if files is None:
            continue
        scenario_files = []
        for f in files:
            if f.get("type") != "file" or f["name"] in _SKIP_FILES:
                continue
            if not f["name"].endswith(".py"):
                continue
            scenario_files.append(
                {
                    "file": f["name"],
                    "resource": f"baybe://recipes/{topic}/{f['name']}",
                    "source": "docs",
                }
            )
        if scenario_files:
            topics[topic] = scenario_files

    return {"baybe_version": version, "topics": topics}


def fetch_recipe(topic: str, filename: str) -> str | None:
    """Fetch the raw content of a single recipe file for the installed version."""
    version = get_baybe_version()
    if version is None:
        return None
    return fetch_text(f"{_RAW}/{version}/examples/{topic}/{filename}")
