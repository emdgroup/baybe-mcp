"""Tests that resource tools return the same content as their resources.

These guard against drift between the tool and resource endpoints.
"""

from __future__ import annotations

import json

from baybe_mcp import server


class TestToolResourceParity:
    def test_list_types_matches_resource(self):
        assert server.list_types() == server.types_resource()

    def test_get_schema_matches_resource(self):
        assert server.get_schema("NumericalDiscreteParameter") == (
            server.schema_resource("NumericalDiscreteParameter")
        )

    def test_get_docs_links_matches_resource(self):
        assert server.get_docs_links() == server.docs_resource()

    def test_list_concepts_matches_resource(self, monkeypatch):
        import baybe_mcp.concepts as concepts

        monkeypatch.setattr(concepts, "fetch_text", lambda *a, **k: None)
        assert server.list_concepts() == server.concepts_index_resource()

    def test_list_recipes_matches_resource(self, monkeypatch):
        import baybe_mcp.recipes as recipes

        monkeypatch.setattr(recipes, "fetch_text", lambda *a, **k: None)
        assert server.list_recipes() == server.recipes_index_resource()


class TestResourceToolsContent:
    def test_get_schema_unknown_type(self):
        result = json.loads(server.get_schema("NoSuchType"))
        assert "error" in result

    def test_get_recipe_offline(self, monkeypatch, tmp_path):
        import baybe_mcp.recipes as recipes

        monkeypatch.setattr(server, "_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(server, "_RECIPES_DIR", str(tmp_path / "recipes"))
        monkeypatch.setattr(recipes, "fetch_text", lambda *a, **k: None)
        result = json.loads(server.get_recipe("Serialization", "validate_config.py"))
        assert "error" in result

    def test_get_recipe_user(self, monkeypatch, tmp_path):
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "howto.md").write_text("# How To")

        monkeypatch.setattr(server, "_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(server, "_RECIPES_DIR", str(recipes_dir))
        assert server.get_recipe("Custom_Recipes", "howto.md") == "# How To"
