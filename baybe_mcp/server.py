"""BayBE MCP Server - exposes BayBE's Bayesian optimization as tools for AI agents."""

from __future__ import annotations

import importlib
import json
import pkgutil
from typing import Any

import attrs
import baybe
import pandas as pd
from baybe.insights.shap import NON_SHAP_EXPLAINERS, SHAP_EXPLAINERS
from baybe.serialization.core import converter
from mcp.server.fastmcp import FastMCP

# Valid SHAP explainer names, derived from the loaded BayBE version so the tool
# description always matches what is actually available at runtime.
_ALL_EXPLAINERS = sorted(SHAP_EXPLAINERS | NON_SHAP_EXPLAINERS)

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


def _as_obj(value: Any) -> Any:
    """Normalize a JSON argument to a native Python object.

    Some MCP clients (e.g. OpenCode) auto-deserialize arguments that are valid
    JSON, so a value may arrive either as a JSON string or as an already-parsed
    dict/list. This returns the parsed object in both cases.

    Already-parsed inputs are deep-copied so that objects owned by the tool
    framework are never passed into BayBE's converter, which would otherwise
    retain references and corrupt later structuring calls.
    """
    if isinstance(value, str):
        return json.loads(value)
    import copy

    return copy.deepcopy(value)


def _deserialize_dataframe(raw: Any) -> pd.DataFrame:
    """Deserialize a DataFrame from JSON input.

    Accepts either a JSON string or an already-parsed object (see ``_as_obj``),
    in three formats:
    - base64: a base64-encoded pickle string (BayBE native)
    - constructor dict: {"constructor": "from_records", "data": [...]}
    - records: a plain JSON array [{"col": val}, ...]
    """
    # A raw base64 string is itself the payload; only parse actual JSON strings.
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped[:1] in ("[", "{", '"'):
            parsed: Any = json.loads(raw)
        else:
            parsed = raw  # base64 payload
    else:
        parsed = raw

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


def _prepare_surrogate(surrogate: Any, objective: Any) -> Any:
    """Ensure the surrogate can model the objective's targets.

    A single-output surrogate cannot model an objective that requires multiple
    models (a ``ParetoObjective``, or a ``DesirabilityObjective`` with
    ``as_pre_transformation=False``): its ``fit`` is rejected and/or its
    ``posterior_stats`` fails to shape the result. Replicating the surrogate
    yields a ``CompositeSurrogate`` with one sub-model per target, which fits
    and computes posterior statistics correctly. A surrogate that already
    handles multiple targets -- one that declares multi-output support, or one
    that has no ``replicate`` method such as a ``CompositeSurrogate`` -- is left
    untouched.

    MAINTENANCE: This is a workaround. If BayBE learns to handle multi-model
    objectives for single-output surrogates directly upstream, remove this and
    pass the surrogate through unchanged.
    """
    if (
        getattr(objective, "_is_multi_model", False)
        and not getattr(surrogate, "supports_multi_output", False)
        and hasattr(surrogate, "replicate")
    ):
        surrogate = surrogate.replicate()
    return surrogate


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
            "serialization": f"{base}/concepts/serialization.html",
            "getting_recommendations": (
                f"{base}/concepts/getting_recommendations.html"
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


def _concepts_index_payload() -> str:
    """Concepts index, from cache if available, else computed live."""
    cached = _read_cached_json("concepts_index.json")
    if cached is not None:
        return json.dumps(cached)

    from baybe_mcp.concepts import build_concepts_index

    return json.dumps(build_concepts_index())


def _concept_payload(name: str) -> str:
    """Raw content of a single concept page; fetched lazily and cached."""
    from baybe_mcp.cache import resolve_cache_dir

    cache_path = resolve_cache_dir(_CACHE_DIR) / f"concepts/{name}.md"
    if cache_path.is_file():
        return cache_path.read_text()

    from baybe_mcp.concepts import fetch_concept

    content = fetch_concept(name)
    if content is None:
        return json.dumps({"error": f"Could not fetch concept '{name}' (offline?)."})

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


@mcp.resource("baybe://concepts")
def concepts_index_resource() -> str:
    """Return the index of BayBE concept explanation pages."""
    return _concepts_index_payload()


@mcp.resource("baybe://concepts/{name}")
def concept_resource(name: str) -> str:
    """Return the raw Markdown of a single BayBE concept page.

    Fetched lazily and cached on first access.
    """
    return _concept_payload(name)


def _build_concepts(cache_dir) -> None:
    """Cache builder for the concepts index (pages fetched lazily)."""
    from baybe_mcp.concepts import build_concepts_index

    (cache_dir / "concepts_index.json").write_text(
        json.dumps(build_concepts_index(), indent=2)
    )


@mcp.tool()
def validate(json_config: str | dict) -> str:
    """Validate a configuration for a BayBE object.

    Validate every config (search space, objective, recommender) with this tool
    BEFORE passing it to `recommend`. It catches errors early and cheaply and
    returns an actionable message you can use to fix the config.

    Accepts either a JSON object or a JSON string. The config must contain a
    "type" field identifying the concrete class (e.g. "SearchSpace",
    "CategoricalParameter", "SingleTargetObjective"). Use `get_schema` to learn
    a type's fields before building the config.

    Returns a JSON object with:
    - "valid": boolean indicating if the config is valid
    - "message": success confirmation or error details to guide a fix
    """
    try:
        data = _as_obj(json_config)
    except json.JSONDecodeError as exc:
        return json.dumps({"valid": False, "message": f"Invalid JSON: {exc}"})

    if not isinstance(data, dict):
        return json.dumps({"valid": False, "message": "Config must be a JSON object."})

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
    searchspace_json: str | dict,
    objective_json: str | dict,
    measurements_json: str | list | dict | None = None,
    recommender_json: str | dict | None = None,
    pending_experiments_json: str | list | dict | None = None,
    output_format: str = "records",
) -> str:
    """Get stateless Bayesian optimization recommendations.

    Performs a single recommendation step without maintaining any server-side
    state (no Campaign object). All context must be passed explicitly.

    Before calling this, build each config using `get_schema` (and
    `get_concept` / `get_example` for patterns), then confirm each
    with `validate`. If this returns an {"error": ...}, re-check the offending
    config with `validate` or `get_schema` and retry.

    Config and measurement arguments accept either a JSON object/array or a
    JSON string; both forms are handled transparently.

    Args:
        batch_size: Number of experiments to recommend.
        searchspace_json: BayBE SearchSpace as a JSON object or string.
        objective_json: BayBE Objective as a JSON object or string.
        measurements_json: Optional past measurements as a DataFrame. Accepts
            records (a JSON array [{"col": val}, ...]), a constructor dict, or a
            base64 string (BayBE native).
        recommender_json: Optional recommender config as a JSON object or
            string. Defaults to TwoPhaseMetaRecommender.
        pending_experiments_json: Optional pending experiments as a DataFrame
            (same formats as measurements).
        output_format: Output format for recommendations: "records"
            (default, list of dicts) or "base64" (BayBE native).

    Returns:
        JSON string with recommended experiments or error details.
    """
    from baybe.objectives.base import Objective
    from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
    from baybe.searchspace.core import SearchSpace

    try:
        searchspace = converter.structure(_as_obj(searchspace_json), SearchSpace)
        objective = converter.structure(_as_obj(objective_json), Objective)

        measurements = None
        if measurements_json is not None:
            measurements = _deserialize_dataframe(measurements_json)

        pending_experiments = None
        if pending_experiments_json is not None:
            pending_experiments = _deserialize_dataframe(pending_experiments_json)

        if recommender_json is not None:
            rec_data = _as_obj(recommender_json)
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


@mcp.tool()
def predict(
    searchspace_json: str | dict,
    objective_json: str | dict,
    candidates_json: str | list | dict,
    measurements_json: str | list | dict,
    stats: list | str | None = None,
    surrogate_json: str | dict | None = None,
    output_format: str = "records",
) -> str:
    """Get posterior statistics (predictions and uncertainty) for candidates.

    Fits a surrogate model on the provided measurements and returns posterior
    statistics for each candidate point, without maintaining any server-side
    state (no Campaign object). All context must be passed explicitly.

    Before calling this, build each config using `get_schema` (and
    `get_concept` / `get_example` for patterns), then confirm each
    with `validate`. If this returns an {"error": ...}, re-check the offending
    config with `validate` or `get_schema` and retry.

    Config, candidate, and measurement arguments accept either a JSON
    object/array or a JSON string; both forms are handled transparently.

    Args:
        searchspace_json: BayBE SearchSpace as a JSON object or string.
        objective_json: BayBE Objective as a JSON object or string.
        candidates_json: Candidate points to predict for, as a DataFrame. Uses
            the same formats as measurements: records (a JSON array
            [{"col": val}, ...]), a constructor dict, or a base64 string (BayBE
            native). Only parameter columns are required (no target values).
        measurements_json: Past measurements used to train the surrogate, as a
            DataFrame (same formats as candidates). Required: the surrogate
            cannot be trained without data.
        stats: Which statistics to compute, as a JSON array or string. Accepts
            "mean", "std", "var", "mode", and floats in the open interval (0, 1)
            for quantiles (e.g. [0.05, 0.95]). Defaults to ["mean", "std"].
        surrogate_json: Optional surrogate config as a JSON object or string.
            Defaults to GaussianProcessSurrogate. For multi-target objectives
            (a ParetoObjective, or a DesirabilityObjective without
            pre-transformation), a single-output surrogate is automatically
            replicated per target, yielding per-target output columns.
        output_format: Output format for statistics: "records" (default, list
            of dicts) or "base64" (BayBE native).

    Returns:
        JSON string with a DataFrame of posterior statistics per candidate
        (columns like "<target>_mean", "<target>_std", "<target>_Q_0.05"), or
        error details.
    """
    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace
    from baybe.surrogates.gaussian_process import GaussianProcessSurrogate

    try:
        searchspace = converter.structure(_as_obj(searchspace_json), SearchSpace)
        objective = converter.structure(_as_obj(objective_json), Objective)

        candidates = _deserialize_dataframe(candidates_json)

        measurements = _deserialize_dataframe(measurements_json)
        if measurements.empty:
            return json.dumps(
                {"error": "measurements are required to train the surrogate."}
            )

        if stats is None:
            requested_stats: Any = ["mean", "std"]
        else:
            requested_stats = _as_obj(stats)
            if not isinstance(requested_stats, list):
                requested_stats = [requested_stats]

        if surrogate_json is not None:
            surr_data = _as_obj(surrogate_json)
            type_name = surr_data.get("type")
            if type_name is None:
                return json.dumps(
                    {"error": 'Missing "type" field in surrogate config.'}
                )
            cls = NAME_TO_CLASS.get(type_name)
            if cls is None:
                return json.dumps({"error": f"Unknown surrogate type: '{type_name}'"})
            surrogate = converter.structure(surr_data, cls)
        else:
            surrogate = GaussianProcessSurrogate()

        surrogate = _prepare_surrogate(surrogate, objective)
        surrogate.fit(searchspace, objective, measurements)
        result = surrogate.posterior_stats(candidates, stats=requested_stats)

        return _serialize_dataframe(result, output_format)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


_PARAMETER_IMPORTANCE_DESCRIPTION = f"""\
Get SHAP-based parameter importance for each target.

Fits a surrogate on the provided measurements, then uses SHAP to attribute the \
model output to each search space parameter. Importance is the mean absolute \
SHAP value of a parameter over the measurements. Stateless: no Campaign or \
server-side state is kept.

Before calling this, build each config using `get_schema` (and \
`get_concept` / `get_example` for patterns), then confirm each with \
`validate`. If this returns an {{"error": ...}}, re-check the offending config \
with `validate` or `get_schema` and retry.

Config and measurement arguments accept either a JSON object/array or a JSON \
string; both forms are handled transparently.

Valid explainer values: {", ".join(_ALL_EXPLAINERS)}. Only KernelExplainer \
supports categorical parameters in the experimental representation \
(use_comp_rep=False). All other explainers require use_comp_rep=True when the \
search space contains categorical parameters; with no categorical parameters, \
any explainer works with use_comp_rep=False.

Args:
    searchspace_json: BayBE SearchSpace as a JSON object or string.
    objective_json: BayBE Objective as a JSON object or string.
    measurements_json: Past measurements used to train the surrogate and as
        SHAP background data, as a DataFrame. Accepts records (a JSON array
        [{{"col": val}}, ...]), a constructor dict, or a base64 string (BayBE
        native). Required: the surrogate cannot be trained without data.
    surrogate_json: Optional surrogate config as a JSON object or string.
        Defaults to GaussianProcessSurrogate. For multi-target objectives
        (a ParetoObjective, or a DesirabilityObjective without
        pre-transformation), a single-output surrogate is automatically
        replicated per target.
    explainer: SHAP explainer class name (default "KernelExplainer"). See the
        valid values listed above.
    use_comp_rep: Explain the computational representation instead of the
        experimental one. Defaults to False.
    output_format: Output format: "records" (default, list of dicts) or
        "base64" (BayBE native).

Returns:
    JSON string with a DataFrame of importances, one row per parameter: a
    "parameter" column plus one "<target>_importance" column per target. Or
    error details.
"""


@mcp.tool(description=_PARAMETER_IMPORTANCE_DESCRIPTION)
def parameter_importance(
    searchspace_json: str | dict,
    objective_json: str | dict,
    measurements_json: str | list | dict,
    surrogate_json: str | dict | None = None,
    explainer: str = "KernelExplainer",
    use_comp_rep: bool = False,
    output_format: str = "records",
) -> str:
    """Get SHAP-based parameter importance for each target.

    The full tool description, including the valid explainer names for the
    installed BayBE version, is built dynamically in
    ``_PARAMETER_IMPORTANCE_DESCRIPTION`` and surfaced to MCP clients.
    """
    import numpy as np
    from baybe.insights.shap import SHAPInsight
    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace
    from baybe.surrogates.gaussian_process import GaussianProcessSurrogate

    try:
        searchspace = converter.structure(_as_obj(searchspace_json), SearchSpace)
        objective = converter.structure(_as_obj(objective_json), Objective)

        measurements = _deserialize_dataframe(measurements_json)
        if measurements.empty:
            return json.dumps(
                {"error": "measurements are required to train the surrogate."}
            )

        if surrogate_json is not None:
            surr_data = _as_obj(surrogate_json)
            type_name = surr_data.get("type")
            if type_name is None:
                return json.dumps(
                    {"error": 'Missing "type" field in surrogate config.'}
                )
            cls = NAME_TO_CLASS.get(type_name)
            if cls is None:
                return json.dumps({"error": f"Unknown surrogate type: '{type_name}'"})
            surrogate = converter.structure(surr_data, cls)
        else:
            surrogate = GaussianProcessSurrogate()

        surrogate = _prepare_surrogate(surrogate, objective)
        surrogate.fit(searchspace, objective, measurements)

        background = measurements[[p.name for p in searchspace.parameters]]
        if use_comp_rep:
            background = searchspace.transform(background)

        insight = SHAPInsight.from_surrogate(
            surrogate, background, explainer_cls=explainer, use_comp_rep=use_comp_rep
        )
        explanations = insight.explain()
        target_names = objective._modeled_quantity_names

        result = None
        for name, explanation in zip(target_names, explanations):
            importance = np.abs(explanation.values).mean(axis=0)
            df = pd.DataFrame(
                {
                    "parameter": list(explanation.feature_names),
                    f"{name}_importance": importance,
                }
            )
            result = df if result is None else result.merge(df, on="parameter")

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

    Step 1 of building a config: discover which "type" values are valid. Each
    entry includes the type name and the URI of its schema. Next, call
    `get_schema` for a type to learn its fields.
    """
    return _types_payload()


@mcp.tool()
def get_schema(type_name: str) -> str:
    """Get the schema for a BayBE type to learn how to build its JSON config.

    Returns the attribute fields (name, type, default, required), alternative
    "from_*" constructors, and docstring. Nested BayBE objects are shown as a
    "$ref" pointing to that type's schema (fetch it with `get_schema`).

    Workflow: discover types with `list_types`, read the schema here, build the
    config, then confirm it with `validate` before calling `recommend`. For
    conventions the schema does not convey (constructor forms, string
    shortcuts, abbreviations), see `get_concept("serialization")`.

    Args:
        type_name: The concrete BayBE type name (e.g. "NumericalDiscreteParameter").
    """
    return _schema_payload(type_name)


@mcp.tool()
def list_examples() -> str:
    """List BayBE example scenarios, grouped by topic.

    Use these for complete, working modelling scenarios (assembling a whole
    search space + objective + recommender), complementing the per-type
    `get_schema`. Fetch a specific file with `get_example`.
    """
    return _examples_index_payload()


@mcp.tool()
def get_example(topic: str, filename: str) -> str:
    """Get the raw content of a single BayBE example scenario file.

    Returns a comprehensive, working modelling scenario. Discover available
    files with `list_examples` first.

    Args:
        topic: The example topic folder (e.g. "Serialization").
        filename: The example file name (e.g. "validate_config.py").
    """
    return _example_file_payload(topic, filename)


@mcp.tool()
def list_concepts() -> str:
    """List BayBE concept explanation pages.

    Concepts explain how BayBE works (e.g. getting recommendations,
    serialization, transfer learning, active learning). Study relevant concepts
    before modelling a project. Fetch a page with `get_concept`.
    """
    return _concepts_index_payload()


@mcp.tool()
def get_concept(name: str) -> str:
    """Get the raw Markdown of a single BayBE concept page.

    Discover available pages with `list_concepts` first. Read
    `get_concept("serialization")` to learn how objects are represented as JSON
    before building configs.

    Args:
        name: The concept page name (e.g. "serialization", "transfer_learning").
    """
    return _concept_payload(name)


@mcp.tool()
def get_docs_links() -> str:
    """Get version-matched links to the BayBE documentation.

    A fallback for deeper reference beyond what `get_schema`, the concept tools,
    and the example tools provide.
    """
    return _docs_payload()


# ---------------------------------------------------------------------------
# Builder registration
# ---------------------------------------------------------------------------

from baybe_mcp.resources import register_builder  # noqa: E402

register_builder("types", _build_types)
register_builder("schema", _build_schema)
register_builder("docs", _build_docs)
register_builder("examples", _build_examples)
register_builder("concepts", _build_concepts)


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
