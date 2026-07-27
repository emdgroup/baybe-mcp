"""Introspection of BayBE classes for building resource content.

Everything here is derived from the installed BayBE package, so the output
always matches the pinned version.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

import attrs

logger = logging.getLogger(__name__)


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
