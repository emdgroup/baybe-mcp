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


# ---------------------------------------------------------------------------
# Resource cache access
# ---------------------------------------------------------------------------


def _read_cached_json(filename: str):
    """Return parsed JSON from the active cache dir, or None if unavailable."""
    from baybe_mcp.cache import resolve_cache_dir

    path = resolve_cache_dir(_CACHE_DIR) / filename
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# Cache dir used by resources at serve time; set during startup.
_CACHE_DIR: str | None = None


# ---------------------------------------------------------------------------
# Payload helpers
#
# Each helper is the single source of truth for one piece of resource content,
# including cache-first behavior. Both the MCP resource and its tool wrapper
# delegate to the same helper, so the two can never drift apart.
# ---------------------------------------------------------------------------


def _types_payload() -> str:
    """Types tree, from cache if available, else computed live."""
    cached = _read_cached_json("types.json")
    if cached is not None:
        return json.dumps(cached)

    from baybe_mcp.introspect import build_types_tree

    return json.dumps(build_types_tree())


def _schema_payload(type_name: str) -> str:
    """Schema for a type, from cache if available, else computed live."""
    cached = _read_cached_json(f"schema/{type_name}.json")
    if cached is not None:
        return json.dumps(cached)

    from baybe_mcp.introspect import build_schema

    return json.dumps(build_schema(type_name))


def _docs_links() -> dict:
    """Build version-matched links to the BayBE documentation."""
    from baybe_mcp.version import docs_base_url, get_baybe_version

    version = get_baybe_version()
    base = docs_base_url(version)
    return {
        "baybe_version": version,
        "links": {
            "user_guide": f"{base}/userguide/userguide.html",
            "serialization": f"{base}/userguide/serialization.html",
            "getting_recommendations": (
                f"{base}/userguide/getting_recommendations.html"
            ),
            "examples": f"{base}/examples/examples.html",
            "api": f"{base}/_autosummary/baybe.html",
        },
    }


def _docs_payload() -> str:
    """Documentation links, from cache if available, else computed live."""
    cached = _read_cached_json("docs.json")
    if cached is not None:
        return json.dumps(cached)
    return json.dumps(_docs_links())


def _guide_payload() -> str:
    """Serialization guide, from cache if available, else fetched live."""
    cached = _read_cached_json("guide_serialization.json")
    if cached is not None:
        return json.dumps(cached)

    from baybe_mcp.guide import build_guide

    return json.dumps(build_guide())


def _examples_index_payload() -> str:
    """Examples index, from cache if available, else computed live."""
    cached = _read_cached_json("examples_index.json")
    if cached is not None:
        return json.dumps(cached)

    from baybe_mcp.examples import build_examples_index

    return json.dumps(build_examples_index())


def _example_file_payload(topic: str, filename: str) -> str:
    """Raw content of a single example file; fetched lazily and cached."""
    from baybe_mcp.cache import resolve_cache_dir

    cache_path = resolve_cache_dir(_CACHE_DIR) / f"examples/{topic}/{filename}"
    if cache_path.is_file():
        return cache_path.read_text()

    from baybe_mcp.examples import fetch_example

    content = fetch_example(topic, filename)
    if content is None:
        return json.dumps(
            {"error": f"Could not fetch example {topic}/{filename} (offline?)."}
        )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content)
    except OSError:
        pass
    return content


# ---------------------------------------------------------------------------
# Resources (delegate to payload helpers)
# ---------------------------------------------------------------------------


@mcp.resource("baybe://types")
def types_resource() -> str:
    """List all serializable BayBE types, grouped by family.

    Each entry includes the type name and the URI of its schema resource.
    """
    return _types_payload()


def _build_types(cache_dir) -> None:
    """Cache builder for baybe://types."""
    from baybe_mcp.introspect import build_types_tree

    (cache_dir / "types.json").write_text(json.dumps(build_types_tree(), indent=2))


@mcp.resource("baybe://schema/{type_name}")
def schema_resource(type_name: str) -> str:
    """Return the schema for a BayBE type.

    Includes attribute fields, alternative constructors, and docstring. Nested
    BayBE-object fields carry a ``$ref`` to their own schema resource.
    """
    return _schema_payload(type_name)


def _build_schema(cache_dir) -> None:
    """Cache builder for baybe://schema/{type}."""
    from baybe_mcp.introspect import build_schema, discover_baybe_classes

    name_to_class = discover_baybe_classes()
    schema_dir = cache_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for type_name in name_to_class:
        schema = build_schema(type_name, name_to_class)
        (schema_dir / f"{type_name}.json").write_text(json.dumps(schema, indent=2))


@mcp.resource("baybe://docs")
def docs_resource() -> str:
    """Return version-matched links to the BayBE documentation."""
    return _docs_payload()


def _build_docs(cache_dir) -> None:
    """Cache builder for baybe://docs."""
    (cache_dir / "docs.json").write_text(json.dumps(_docs_links(), indent=2))


@mcp.resource("baybe://guide/serialization")
def serialization_guide_resource() -> str:
    """Return the BayBE serialization guide for the installed version.

    Served from cache if available; otherwise fetched live, with a link
    fallback when offline.
    """
    return _guide_payload()


def _build_guide(cache_dir) -> None:
    """Cache builder for baybe://guide/serialization."""
    from baybe_mcp.guide import build_guide

    (cache_dir / "guide_serialization.json").write_text(
        json.dumps(build_guide(), indent=2)
    )


@mcp.resource("baybe://examples")
def examples_index_resource() -> str:
    """Return the index of BayBE example scenarios (topics and files)."""
    return _examples_index_payload()


@mcp.resource("baybe://examples/{topic}/{filename}")
def example_file_resource(topic: str, filename: str) -> str:
    """Return the raw content of a single example scenario file.

    Fetched lazily and cached on first access.
    """
    return _example_file_payload(topic, filename)


def _build_examples(cache_dir) -> None:
    """Cache builder for the examples index (files fetched lazily)."""
    from baybe_mcp.examples import build_examples_index

    (cache_dir / "examples_index.json").write_text(
        json.dumps(build_examples_index(), indent=2)
    )


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
# Resource tools
#
# The same content exposed as MCP resources is also exposed as tools, because
# some MCP clients surface only tools (not resources) to the agent. Each tool
# delegates to the same payload helper as its resource.
# ---------------------------------------------------------------------------


@mcp.tool()
def list_types() -> str:
    """List all serializable BayBE types, grouped by family.

    Use this to discover which "type" values are valid. Each entry includes the
    type name and the URI of its schema (retrievable with get_schema).
    """
    return _types_payload()


@mcp.tool()
def get_schema(type_name: str) -> str:
    """Get the schema for a BayBE type to learn how to build its JSON config.

    Returns the attribute fields (name, type, default, required), alternative
    "from_*" constructors, and docstring. Nested BayBE objects are shown as a
    "$ref" pointing to that type's schema. Call this before writing a config.

    Args:
        type_name: The concrete BayBE type name (e.g. "NumericalDiscreteParameter").
    """
    return _schema_payload(type_name)


@mcp.tool()
def get_serialization_guide() -> str:
    """Get the BayBE serialization guide for the installed version.

    Explains conventions that schemas alone do not convey (alternative
    constructors, string shortcuts, abbreviations, dataframe formats).
    """
    return _guide_payload()


@mcp.tool()
def list_examples() -> str:
    """List BayBE example scenarios, grouped by topic.

    Each file can be fetched with get_example to see a comprehensive, working
    modelling scenario.
    """
    return _examples_index_payload()


@mcp.tool()
def get_example(topic: str, filename: str) -> str:
    """Get the raw content of a single BayBE example scenario file.

    Args:
        topic: The example topic folder (e.g. "Serialization").
        filename: The example file name (e.g. "validate_config.py").
    """
    return _example_file_payload(topic, filename)


@mcp.tool()
def get_docs_links() -> str:
    """Get version-matched links to the BayBE documentation."""
    return _docs_payload()


# ---------------------------------------------------------------------------
# Builder registration
# ---------------------------------------------------------------------------

from baybe_mcp.resources import register_builder  # noqa: E402

register_builder("types", _build_types)
register_builder("schema", _build_schema)
register_builder("docs", _build_docs)
register_builder("guide", _build_guide)
register_builder("examples", _build_examples)


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

    global _CACHE_DIR
    _CACHE_DIR = args.cache_dir

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
