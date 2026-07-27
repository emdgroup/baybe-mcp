"""Detection of the installed BayBE version and derivation of doc URLs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DOCS_BASE = "https://emdgroup.github.io/baybe"


def get_baybe_version() -> str | None:
    """Return the installed BayBE version, or None if it cannot be detected.

    A warning is logged when detection fails.
    """
    try:
        import baybe

        version = getattr(baybe, "__version__", None)
        if not version or not isinstance(version, str):
            logger.warning("Could not detect BayBE version from baybe.__version__.")
            return None
        return version
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not import baybe to detect version: %s", exc)
        return None


def docs_base_url(version: str | None = None) -> str:
    """Return the versioned BayBE docs base URL.

    Falls back to the "latest" docs (with a warning) when no version is given
    or detectable.
    """
    if version is None:
        version = get_baybe_version()
    if version is None:
        logger.warning("Linking to 'latest' BayBE docs (version undetectable).")
        return f"{DOCS_BASE}/latest"
    return f"{DOCS_BASE}/{version}"
