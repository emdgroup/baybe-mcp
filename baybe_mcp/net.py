"""Network helpers: availability probe and simple fetch."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Host used to check whether the BayBE docs/repo are reachable.
PROBE_URL = "https://raw.githubusercontent.com/emdgroup/baybe/main/README.md"


def is_online(timeout: float = 5.0, url: str = PROBE_URL) -> bool:
    """Return True if the given URL is reachable within the timeout."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def fetch_text(url: str, timeout: float = 10.0) -> str | None:
    """Fetch text content from a URL, or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if 200 <= resp.status < 400:
                return resp.read().decode("utf-8")
            return None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None
