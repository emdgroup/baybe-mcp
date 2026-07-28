"""Tests for version helpers, cache, and resources (network mocked)."""

from __future__ import annotations

import json

from baybe_mcp import cache, introspect, recipes, version


# ---------------------------------------------------------------------------
# version helpers
# ---------------------------------------------------------------------------


class TestVersion:
    def test_detects_installed_version(self):
        v = version.get_baybe_version()
        assert v is not None
        assert v.count(".") >= 2  # e.g. 0.15.0

    def test_docs_url_contains_version(self):
        v = version.get_baybe_version()
        assert version.docs_base_url(v).endswith(v)

    def test_docs_url_latest_fallback(self, monkeypatch):
        # When the version cannot be detected, fall back to 'latest'.
        monkeypatch.setattr(version, "get_baybe_version", lambda: None)
        assert version.docs_base_url(None).endswith("latest")


# ---------------------------------------------------------------------------
# cache version-guard
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_invalid_before_build(self, tmp_path):
        assert cache.is_cache_valid(tmp_path) is False

    def test_cache_valid_after_matching_build(self, tmp_path):
        cache.write_manifest(tmp_path, version.get_baybe_version())
        assert cache.is_cache_valid(tmp_path) is True

    def test_cache_invalid_on_version_mismatch(self, tmp_path):
        cache.write_manifest(tmp_path, "0.0.0-not-installed")
        assert cache.is_cache_valid(tmp_path) is False


# ---------------------------------------------------------------------------
# introspection: types + schema
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_types_tree_grouped(self):
        tree = introspect.build_types_tree()
        assert "Parameter" in tree
        names = {e["type"] for e in tree["Parameter"]}
        assert "NumericalDiscreteParameter" in names

    def test_schema_has_fields(self):
        schema = introspect.build_schema("NumericalDiscreteParameter")
        assert "values" in schema["fields"]
        assert schema["fields"]["name"]["required"] is True

    def test_schema_lists_alternative_constructors(self):
        schema = introspect.build_schema("SearchSpace")
        assert "from_product" in schema["constructors"]
        assert "from_json" not in schema["constructors"]

    def test_schema_nested_ref(self):
        schema = introspect.build_schema("SingleTargetObjective")
        assert schema["fields"]["target"]["$ref"] == "baybe://schema/Target"


# ---------------------------------------------------------------------------
# recipes (network mocked)
# ---------------------------------------------------------------------------


class TestRecipes:
    def test_index_success(self, monkeypatch):
        def fake_fetch(url, timeout=10.0):
            if url.endswith("examples?ref=" + (version.get_baybe_version() or "")):
                return json.dumps([{"name": "Basics", "type": "dir"}])
            if "examples/Basics" in url:
                return json.dumps([{"name": "start.py", "type": "file"}])
            return None

        monkeypatch.setattr(recipes, "fetch_text", fake_fetch)
        idx = recipes.build_recipes_index()
        assert "Basics" in idx["topics"]
        assert idx["topics"]["Basics"][0]["file"] == "start.py"
        assert idx["topics"]["Basics"][0]["source"] == "docs"

    def test_index_offline_fallback(self, monkeypatch):
        monkeypatch.setattr(recipes, "fetch_text", lambda *a, **k: None)
        idx = recipes.build_recipes_index()
        assert idx["topics"] == {}
        assert "link" in idx
