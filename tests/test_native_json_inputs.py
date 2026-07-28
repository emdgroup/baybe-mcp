"""Tests that validate/recommend accept native JSON (dict/list) inputs.

Some MCP clients (e.g. OpenCode) auto-deserialize valid-JSON arguments, so the
tools must handle already-parsed dict/list values, not just JSON strings. These
tests exercise the tools through the MCP machinery, which mirrors that path.
"""

from __future__ import annotations

import json

import pytest

from baybe_mcp.server import mcp

SEARCHSPACE = {
    "type": "SearchSpace",
    "constructor": "from_product",
    "parameters": [
        {"type": "NumericalDiscreteParameter", "name": "x1", "values": [1.0, 2.0, 3.0]},
        {"type": "NumericalDiscreteParameter", "name": "x2", "values": [10.0, 20.0]},
    ],
}
OBJECTIVE = {
    "type": "SingleTargetObjective",
    "target": {
        "type": "NumericalTarget",
        "name": "y",
        "transformation": {"type": "IdentityTransformation"},
        "minimize": False,
    },
}
MEASUREMENTS = [
    {"x1": 1.0, "x2": 10.0, "y": 0.5},
    {"x1": 3.0, "x2": 20.0, "y": 0.8},
]
CANDIDATES = [
    {"x1": 2.0, "x2": 10.0},
    {"x1": 2.0, "x2": 20.0},
]


async def _call(name, args):
    _, structured = await mcp.call_tool(name, args)
    return json.loads(structured["result"])


class TestValidateInputs:
    @pytest.mark.asyncio
    async def test_validate_native_dict(self):
        result = await _call("validate", {"json_config": SEARCHSPACE})
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_json_string(self):
        result = await _call("validate", {"json_config": json.dumps(SEARCHSPACE)})
        assert result["valid"] is True


class TestRecommendInputs:
    @pytest.mark.asyncio
    async def test_recommend_native_dict_and_list(self):
        """The OpenCode case: configs as dicts, measurements as a list."""
        result = await _call(
            "recommend",
            {
                "batch_size": 2,
                "searchspace_json": SEARCHSPACE,
                "objective_json": OBJECTIVE,
                "measurements_json": MEASUREMENTS,
            },
        )
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_recommend_json_strings(self):
        """Backward compatibility: stringified JSON still works."""
        result = await _call(
            "recommend",
            {
                "batch_size": 2,
                "searchspace_json": json.dumps(SEARCHSPACE),
                "objective_json": json.dumps(OBJECTIVE),
                "measurements_json": json.dumps(MEASUREMENTS),
            },
        )
        assert isinstance(result, list)
        assert len(result) == 2


class TestPredictInputs:
    @pytest.mark.asyncio
    async def test_predict_native_inputs(self):
        """The OpenCode case: configs/candidates/stats as native objects."""
        result = await _call(
            "predict",
            {
                "searchspace_json": SEARCHSPACE,
                "objective_json": OBJECTIVE,
                "candidates_json": CANDIDATES,
                "measurements_json": MEASUREMENTS,
                "stats": ["mean", 0.05],
            },
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert "y_mean" in result[0]
        assert "y_Q_0.05" in result[0]

    @pytest.mark.asyncio
    async def test_predict_json_strings(self):
        """Backward compatibility: stringified JSON still works."""
        result = await _call(
            "predict",
            {
                "searchspace_json": json.dumps(SEARCHSPACE),
                "objective_json": json.dumps(OBJECTIVE),
                "candidates_json": json.dumps(CANDIDATES),
                "measurements_json": json.dumps(MEASUREMENTS),
            },
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert "y_mean" in result[0]
