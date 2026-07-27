"""Serialization guide resource content, fetched for the installed version."""

from __future__ import annotations

import logging

from baybe_mcp.net import fetch_text
from baybe_mcp.version import docs_base_url, get_baybe_version

logger = logging.getLogger(__name__)

# Candidate raw-Markdown locations for the serialization guide, tried in order.
# The path moved across BayBE versions, so more than one candidate is tried.
_MD_CANDIDATES = (
    "docs/concepts/serialization.md",
    "docs/userguide/serialization.md",
)


def _raw_url(version: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/emdgroup/baybe/{version}/{path}"


def build_guide() -> dict:
    """Fetch the serialization guide for the installed BayBE version.

    Tries the raw Markdown at the version tag first. On failure, falls back to
    a link to the rendered (version-matched or latest) guide page. The returned
    dict always contains a ``link``; ``content`` is present only when fetched.
    """
    version = get_baybe_version()
    link = f"{docs_base_url(version)}/userguide/serialization.html"

    if version is not None:
        for path in _MD_CANDIDATES:
            content = fetch_text(_raw_url(version, path))
            if content is not None:
                return {
                    "baybe_version": version,
                    "source": _raw_url(version, path),
                    "content": content,
                }

    logger.warning(
        "Could not fetch serialization guide; providing a documentation link."
    )
    return {
        "baybe_version": version,
        "link": link,
        "note": "Guide content unavailable offline; see the linked documentation.",
    }
