"""Introspection of BayBE classes for building resource content.

Everything here is derived from the installed BayBE package, so the output
always matches the pinned version.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

import attrs

logger = logging.getLogger(__name__)

# Constructors that are not semantic alternative constructors: the serialization
# round-trip helpers (inherited) and a few niche variants.
_CONSTRUCTOR_BLOCKLIST = {
    "from_json",
    "from_dict",
    "from_constructor_info",
    "from_legacy_interface",
    "from_modern_interface",
}


def discover_baybe_classes() -> dict[str, type]:
    """Walk the baybe package and collect all attrs classes by name."""
    import baybe

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


def _base_classes() -> dict[str, type]:
    """Return the base classes used to group serializable types.

    Missing bases (across BayBE versions) are skipped gracefully.
    """
    specs = [
        ("Parameter", "baybe.parameters.base", "Parameter"),
        ("Target", "baybe.targets.base", "Target"),
        ("Objective", "baybe.objectives.base", "Objective"),
        ("Constraint", "baybe.constraints.base", "Constraint"),
        ("Recommender", "baybe.recommenders.base", "RecommenderProtocol"),
        ("Surrogate", "baybe.surrogates.base", "Surrogate"),
        ("AcquisitionFunction", "baybe.acquisition.base", "AcquisitionFunction"),
        ("Kernel", "baybe.kernels.base", "Kernel"),
        ("Prior", "baybe.priors.base", "Prior"),
    ]
    bases: dict[str, type] = {}
    for label, path, name in specs:
        try:
            mod = importlib.import_module(path)
            bases[label] = getattr(mod, name)
        except Exception:
            continue
    return bases


def build_types_tree(name_to_class: dict[str, type] | None = None) -> dict:
    """Group all serializable types by base-class family.

    Returns a mapping of family label -> list of entries, each carrying the
    type name and its flat schema resource URI. Types not belonging to a known
    family are grouped under "Other".
    """
    if name_to_class is None:
        name_to_class = discover_baybe_classes()
    bases = _base_classes()

    groups: dict[str, list[dict]] = {label: [] for label in bases}
    groups["Other"] = []

    for type_name, cls in sorted(name_to_class.items()):
        entry = {
            "type": type_name,
            "schema": f"baybe://schema/{type_name}",
        }
        placed = False
        for label, base in bases.items():
            try:
                if issubclass(cls, base):
                    groups[label].append(entry)
                    placed = True
                    break
            except TypeError:
                continue
        if not placed:
            groups["Other"].append(entry)

    # Drop empty families for a cleaner tree.
    return {label: entries for label, entries in groups.items() if entries}


def _type_str(tp) -> str:
    """Render a field/parameter type annotation as a readable string."""
    if tp is None:
        return "Any"
    if isinstance(tp, str):
        return tp
    return getattr(tp, "__name__", str(tp))


def _nested_ref(tp, name_to_class: dict[str, type]) -> str | None:
    """Return a schema URI if the annotation refers to a known BayBE type."""
    tname = _type_str(tp)
    # Strip common container wrappers to find an inner known type name.
    for known in name_to_class:
        if known == tname or f"[{known}" in tname or f" {known}" in tname:
            return f"baybe://schema/{known}"
    return None


def _alternative_constructors(cls: type) -> dict:
    """Return class-defined from_* constructors with their signatures."""
    result: dict = {}
    for attr_name in vars(cls):
        if not attr_name.startswith("from_") or attr_name in _CONSTRUCTOR_BLOCKLIST:
            continue
        member = inspect.getattr_static(cls, attr_name)
        if not isinstance(member, classmethod):
            continue
        func = member.__func__
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            continue
        params = {}
        for pname, param in sig.parameters.items():
            if pname == "cls":
                continue
            params[pname] = {
                "type": _type_str(param.annotation)
                if param.annotation is not inspect.Parameter.empty
                else "Any",
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                params[pname]["default"] = repr(param.default)
        result[attr_name] = {
            "doc": inspect.getdoc(func),
            "parameters": params,
        }
    return result


def build_schema(type_name: str, name_to_class: dict[str, type] | None = None) -> dict:
    """Build the schema description for a single BayBE type.

    Includes attribute fields (name, type, default, required), alternative
    ``from_*`` constructors with signatures, and the class docstring. Fields
    whose type is another BayBE object carry a ``$ref`` to that type's schema.
    """
    if name_to_class is None:
        name_to_class = discover_baybe_classes()
    cls = name_to_class.get(type_name)
    if cls is None:
        return {"error": f"Unknown BayBE type: '{type_name}'"}

    fields: dict = {}
    for f in attrs.fields(cls):
        key = f.alias or f.name.lstrip("_")
        entry = {
            "type": _type_str(f.type),
            "required": f.default is attrs.NOTHING,
        }
        if f.default is not attrs.NOTHING:
            entry["default"] = repr(f.default)
        ref = _nested_ref(f.type, name_to_class)
        if ref is not None:
            entry["$ref"] = ref
        fields[key] = entry

    schema = {
        "type": type_name,
        "doc": inspect.getdoc(cls),
        "fields": fields,
        "constructors": _alternative_constructors(cls),
    }

    # SHAPInsight's `explainers` field holds explainer instances, not the string
    # names callers pass to the parameter_importance tool. Surface the valid
    # names explicitly, derived from the loaded BayBE version.
    if type_name == "SHAPInsight":
        try:
            from baybe.insights.shap import EXPLAINERS

            schema["valid_explainers"] = sorted(EXPLAINERS)
        except Exception:
            pass

    return schema
