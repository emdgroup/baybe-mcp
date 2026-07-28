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


# Topic under which top-level user recipe files are grouped.
USER_TOPIC = "Custom_Recipes"


def build_recipes_index(recipes_dir=None) -> dict:
    """Build the index of recipe topics and their scenario files.

    Returns a dict with the version and, per topic, the list of files with
    their resource URIs and source. Doc recipes are fetched from the examples
    folder at the version tag; user recipes (local ``.md`` files) are merged in
    from ``recipes_dir`` under a separate namespace. Falls back to a link when
    the doc listing is unavailable (user recipes are still included).
    """
    version = get_baybe_version()
    result: dict = {"baybe_version": version, "topics": {}}

    if version is None:
        result["note"] = "Version undetectable."
    else:
        entries = _contents(version, "examples")
        if entries is None:
            result["link"] = (
                f"https://github.com/emdgroup/baybe/tree/{version}/examples"
            )
            result["note"] = "Recipe index unavailable offline; see the linked folder."
        else:
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
            result["topics"] = topics

    _merge_user_recipes(result["topics"], recipes_dir)
    return result


def _merge_user_recipes(topics: dict, recipes_dir) -> None:
    """Merge user-provided ``.md`` recipes into the topics mapping in place.

    Subfolders become topics; top-level files are grouped under ``USER_TOPIC``.
    User recipes live in a separate namespace and never override doc recipes:
    if a user topic name collides with a doc topic, user files are appended.
    """
    from baybe_mcp.cache import resolve_recipes_dir

    directory = resolve_recipes_dir(recipes_dir)
    if not directory.is_dir():
        return

    for path in sorted(directory.rglob("*.md")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory)
        topic = rel.parts[0] if len(rel.parts) > 1 else USER_TOPIC
        entry = {
            "file": path.name,
            "resource": f"baybe://recipes/{topic}/{path.name}",
            "source": "user",
        }
        topics.setdefault(topic, []).append(entry)


def fetch_recipe(topic: str, filename: str) -> str | None:
    """Fetch the raw content of a single recipe file for the installed version."""
    version = get_baybe_version()
    if version is None:
        return None
    return fetch_text(f"{_RAW}/{version}/examples/{topic}/{filename}")
