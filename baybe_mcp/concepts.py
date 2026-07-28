"""Concept explanation pages sourced from BayBE's docs/concepts folder.

The index of concept pages is discovered via the GitHub contents API at the
version tag; individual pages are fetched as raw Markdown on demand.
"""

from __future__ import annotations

import json
import logging

from baybe_mcp.net import fetch_text
from baybe_mcp.version import get_baybe_version

logger = logging.getLogger(__name__)

_API = "https://api.github.com/repos/emdgroup/baybe/contents"
_RAW = "https://raw.githubusercontent.com/emdgroup/baybe"

# The landing page is a table of contents, not a concept explanation.
_SKIP_FILES = {"index.md"}


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


def build_concepts_index() -> dict:
    """Build the index of concept explanation pages.

    Returns a dict with the version and the list of concept pages with their
    resource URIs. Falls back to a link when the listing is unavailable.
    """
    version = get_baybe_version()
    if version is None:
        return {"baybe_version": None, "note": "Version undetectable.", "concepts": []}

    entries = _contents(version, "docs/concepts")
    if entries is None:
        return {
            "baybe_version": version,
            "link": f"https://github.com/emdgroup/baybe/tree/{version}/docs/concepts",
            "note": "Concept index unavailable offline; see the linked folder.",
            "concepts": [],
        }

    concepts = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = entry["name"]
        if name in _SKIP_FILES or not name.endswith(".md"):
            continue
        concept = name[: -len(".md")]
        concepts.append(
            {
                "concept": concept,
                "resource": f"baybe://concepts/{concept}",
            }
        )

    return {"baybe_version": version, "concepts": concepts}


def fetch_concept(name: str) -> str | None:
    """Fetch the raw Markdown of a single concept page for the installed version."""
    version = get_baybe_version()
    if version is None:
        return None
    return fetch_text(f"{_RAW}/{version}/docs/concepts/{name}.md")
