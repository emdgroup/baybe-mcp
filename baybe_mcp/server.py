"""BayBE MCP Server - exposes BayBE's Bayesian optimization as tools for AI agents."""

from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Any

import attrs
import baybe
import pandas as pd
from baybe.serialization.core import converter
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Class discovery: build a map of all concrete attrs classes in baybe
# ---------------------------------------------------------------------------


def _discover_baybe_classes() -> dict[str, type]:
    """Walk the baybe package and collect all attrs classes by name."""
    name_to_class: dict[str, type] = {}
    for _, modname, _ in pkgutil.walk_packages(baybe.__path__, prefix="baybe."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if isinstance(obj, type) and attrs.has(obj):
                name_to_class[obj.__name__] = obj
    return name_to_class


NAME_TO_CLASS = _discover_baybe_classes()

# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _deserialize_dataframe(raw: str) -> pd.DataFrame:
    """Deserialize a DataFrame from JSON.

    Supports three formats:
    - base64: a base64-encoded pickle string (BayBE native)
    - constructor dict: {"constructor": "from_records", "data": [...]}
    - records: a plain JSON array [{"col": val}, ...]
    """
    parsed: Any = json.loads(raw)

    # base64 string or constructor dict → let BayBE's converter handle it
    if isinstance(parsed, str) or (
        isinstance(parsed, dict) and "constructor" in parsed
    ):
        return converter.structure(parsed, pd.DataFrame)

    # records format (list of dicts)
    if isinstance(parsed, list):
        return pd.DataFrame(parsed)

    raise ValueError(
        "Unsupported measurements format. Provide a base64 string, "
        'a dict with "constructor" key, or a list of record dicts.'
    )


def _serialize_dataframe(df: pd.DataFrame, output_format: str) -> str:
    """Serialize a DataFrame to the requested format."""
    if output_format == "base64":
        return json.dumps(converter.unstructure(df))
    # default: records
    return json.dumps(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("baybe-mcp")


@mcp.tool()
def validate(json_config: str) -> str:
    """Validate a JSON configuration for a BayBE object.

    Takes a JSON string representing a BayBE object. The JSON must contain a
    "type" field identifying the concrete class (e.g. "SearchSpace",
    "CategoricalParameter", "SingleTargetObjective").

    Returns a JSON object with:
    - "valid": boolean indicating if the config is valid
    - "message": success confirmation or error details
    """
    try:
        data = json.loads(json_config)
    except json.JSONDecodeError as exc:
        return json.dumps({"valid": False, "message": f"Invalid JSON: {exc}"})

    type_name = data.get("type")
    if type_name is None:
        return json.dumps(
            {"valid": False, "message": 'Missing "type" field in config.'}
        )

    cls = NAME_TO_CLASS.get(type_name)
    if cls is None:
        return json.dumps(
            {
                "valid": False,
                "message": f"Unknown BayBE type: '{type_name}'. "
                f"Available types: {sorted(NAME_TO_CLASS.keys())}",
            }
        )

    try:
        converter.structure(data, cls)
    except Exception as exc:
        return json.dumps({"valid": False, "message": f"Validation error: {exc}"})

    return json.dumps({"valid": True, "message": f"Valid {type_name} configuration."})


@mcp.tool()
def recommend(
    batch_size: int,
    searchspace_json: str,
    objective_json: str,
    measurements_json: str | None = None,
    recommender_json: str | None = None,
    pending_experiments_json: str | None = None,
    output_format: str = "records",
) -> str:
    """Get stateless Bayesian optimization recommendations.

    Performs a single recommendation step without maintaining any server-side
    state (no Campaign object). All context must be passed explicitly.

    Args:
        batch_size: Number of experiments to recommend.
        searchspace_json: JSON-serialized BayBE SearchSpace.
        objective_json: JSON-serialized BayBE Objective.
        measurements_json: Optional JSON-serialized DataFrame of past
            measurements. Supports three formats: base64 (BayBE native),
            constructor dict, or records ([{"col": val}, ...]).
        recommender_json: Optional JSON-serialized recommender config.
            Defaults to TwoPhaseMetaRecommender.
        pending_experiments_json: Optional JSON-serialized DataFrame of
            pending experiments (same formats as measurements).
        output_format: Output format for recommendations: "records"
            (default, list of dicts) or "base64" (BayBE native).

    Returns:
        JSON string with recommended experiments or error details.
    """
    from baybe.objectives.base import Objective
    from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
    from baybe.searchspace.core import SearchSpace

    try:
        searchspace = SearchSpace.from_json(searchspace_json)
        objective = Objective.from_json(objective_json)

        measurements = None
        if measurements_json is not None:
            measurements = _deserialize_dataframe(measurements_json)

        pending_experiments = None
        if pending_experiments_json is not None:
            pending_experiments = _deserialize_dataframe(pending_experiments_json)

        if recommender_json is not None:
            rec_data = json.loads(recommender_json)
            type_name = rec_data.get("type")
            if type_name is None:
                return json.dumps(
                    {"error": 'Missing "type" field in recommender config.'}
                )
            cls = NAME_TO_CLASS.get(type_name)
            if cls is None:
                return json.dumps({"error": f"Unknown recommender type: '{type_name}'"})
            recommender = converter.structure(rec_data, cls)
        else:
            recommender = TwoPhaseMetaRecommender()

        result = recommender.recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
            pending_experiments=pending_experiments,
        )

        return _serialize_dataframe(result, output_format)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _ensure_resources(args) -> None:
    """Ensure the resource cache is present and current before serving.

    Logic:
    - ``--rebuild-resources``: always rebuild.
    - valid cache: use it.
    - stale/missing cache: rebuild if online, otherwise warn and serve
      degraded (resources fall back to live introspection / links).
    - ``--use-cache``: require a valid cache; error if stale/missing.
    """
    import logging

    from baybe_mcp.cache import is_cache_valid, resolve_cache_dir
    from baybe_mcp.net import is_online
    from baybe_mcp.resources import build_resources

    logger = logging.getLogger(__name__)
    cache_dir = resolve_cache_dir(args.cache_dir)

    if args.rebuild_resources:
        build_resources(cache_dir=cache_dir)
        return

    if is_cache_valid(cache_dir):
        logger.info("Using cached resources at %s", cache_dir)
        return

    if args.use_cache:
        raise SystemExit(
            f"No valid resource cache at {cache_dir} (built for a different "
            "BayBE version or format). Run the 'build' command first, or drop "
            "--use-cache to allow rebuilding."
        )

    if is_online():
        logger.info("Resource cache stale/missing; rebuilding.")
        build_resources(cache_dir=cache_dir)
    else:
        logger.warning(
            "Resource cache stale/missing and no network available; serving "
            "with degraded resources (live introspection and doc links only)."
        )


def _run_server(args) -> None:
    """Start the MCP server with the given parsed arguments."""
    mcp.settings.log_level = args.log_level
    _ensure_resources(args)
    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


def _build(args) -> None:
    """Build the resource cache as a standalone step."""
    import logging

    from baybe_mcp.resources import build_resources

    logging.basicConfig(level=args.log_level)
    build_resources(cache_dir=args.cache_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``run`` and ``build`` subcommands.

    A bare invocation (no subcommand) defaults to ``run`` for backwards
    compatibility.
    """
    import argparse

    parser = argparse.ArgumentParser(description="BayBE MCP server.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the MCP server.")
    run_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport to use (default: stdio).",
    )
    run_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP transport (default: 127.0.0.1).",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transport (default: 8000).",
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    run_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: .baybe_mcp_cache in the current dir).",
    )
    run_parser.add_argument(
        "--rebuild-resources",
        action="store_true",
        help="Rebuild the resource cache before serving.",
    )
    run_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Require a valid cache; error instead of rebuilding (offline use).",
    )

    build_parser = subparsers.add_parser(
        "build", help="Build the resource cache and exit."
    )
    build_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: .baybe_mcp_cache in the current dir).",
    )
    build_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        _build(args)
        return

    # Default to "run" when no (or the "run") subcommand is given. argparse only
    # populated run-specific args if the run subparser was used, so fall back to
    # a fresh parse of the run defaults for the bare-invocation case.
    if args.command is None:
        args = run_parser.parse_args([])
    _run_server(args)


if __name__ == "__main__":
    main()
