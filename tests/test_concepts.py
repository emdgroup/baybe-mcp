"""Tests for the concepts resource (network mocked)."""

from __future__ import annotations

import json

from baybe_mcp import concepts, version


def test_index_success(monkeypatch):
    def fake_fetch(url, timeout=10.0):
        if url.endswith("docs/concepts?ref=" + (version.get_baybe_version() or "")):
            return json.dumps(
                [
                    {"name": "serialization.md", "type": "file"},
                    {"name": "index.md", "type": "file"},
                    {"name": "subdir", "type": "dir"},
                ]
            )
        return None

    monkeypatch.setattr(concepts, "fetch_text", fake_fetch)
    idx = concepts.build_concepts_index()
    names = {c["concept"] for c in idx["concepts"]}
    assert "serialization" in names
    # index.md and directories are excluded.
    assert "index" not in names
    assert idx["concepts"][0]["resource"].startswith("baybe://concepts/")


def test_index_offline_fallback(monkeypatch):
    monkeypatch.setattr(concepts, "fetch_text", lambda *a, **k: None)
    idx = concepts.build_concepts_index()
    assert idx["concepts"] == []
    assert "link" in idx


def test_fetch_concept(monkeypatch):
    monkeypatch.setattr(concepts, "fetch_text", lambda *a, **k: "# Serialization\n...")
    content = concepts.fetch_concept("serialization")
    assert content.startswith("# Serialization")


def test_fetch_concept_offline(monkeypatch):
    monkeypatch.setattr(concepts, "fetch_text", lambda *a, **k: None)
    assert concepts.fetch_concept("serialization") is None
