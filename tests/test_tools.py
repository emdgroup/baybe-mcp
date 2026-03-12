"""Tests for the BayBE MCP server tools."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.serialization.core import converter
from baybe.targets.numerical import NumericalTarget

from baybe_mcp.server import recommend, validate

# ---------------------------------------------------------------------------
# Fixtures: reusable serialized BayBE objects
# ---------------------------------------------------------------------------

SEARCHSPACE_JSON = json.dumps(
    {
        "type": "SearchSpace",
        "constructor": "from_product",
        "parameters": [
            {
                "type": "NumericalDiscreteParameter",
                "name": "x1",
                "values": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            {
                "type": "NumericalDiscreteParameter",
                "name": "x2",
                "values": [10.0, 20.0, 30.0],
            },
        ],
    }
)

OBJECTIVE_JSON = json.dumps(
    {
        "type": "SingleTargetObjective",
        "target": {
            "type": "NumericalTarget",
            "name": "y",
            "transformation": {"type": "IdentityTransformation"},
            "minimize": False,
        },
    }
)


# ---------------------------------------------------------------------------
# validate tool tests
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_parameter(self):
        config = json.dumps(
            {
                "type": "NumericalDiscreteParameter",
                "name": "temperature",
                "values": [100.0, 150.0, 200.0],
            }
        )
        result = json.loads(validate(config))
        assert result["valid"] is True

    def test_valid_searchspace(self):
        result = json.loads(validate(SEARCHSPACE_JSON))
        assert result["valid"] is True

    def test_valid_objective(self):
        result = json.loads(validate(OBJECTIVE_JSON))
        assert result["valid"] is True

    def test_invalid_json(self):
        result = json.loads(validate("not valid json {{{"))
        assert result["valid"] is False
        assert "Invalid JSON" in result["message"]

    def test_missing_type_field(self):
        config = json.dumps({"name": "x", "values": [1, 2, 3]})
        result = json.loads(validate(config))
        assert result["valid"] is False
        assert "type" in result["message"]

    def test_unknown_type(self):
        config = json.dumps({"type": "NonExistentClass", "name": "x"})
        result = json.loads(validate(config))
        assert result["valid"] is False
        assert "Unknown BayBE type" in result["message"]

    def test_invalid_config_values(self):
        # NumericalDiscreteParameter requires non-empty values
        config = json.dumps(
            {
                "type": "NumericalDiscreteParameter",
                "name": "x",
                "values": [],
            }
        )
        result = json.loads(validate(config))
        assert result["valid"] is False
        assert "message" in result


# ---------------------------------------------------------------------------
# recommend tool tests
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_recommend_without_measurements(self):
        """Initial recommendation (no measurements) should work with default recommender."""
        result_str = recommend(
            batch_size=2,
            searchspace_json=SEARCHSPACE_JSON,
            objective_json=OBJECTIVE_JSON,
        )
        result = json.loads(result_str)
        # Should be a list of dicts (records format)
        assert isinstance(result, list)
        assert len(result) == 2
        # Each record should contain the parameter columns
        assert "x1" in result[0]
        assert "x2" in result[0]

    def test_recommend_with_measurements_records(self):
        """Recommendation with measurements in records format."""
        measurements = json.dumps(
            [
                {"x1": 1.0, "x2": 10.0, "y": 0.5},
                {"x1": 3.0, "x2": 20.0, "y": 0.8},
            ]
        )
        result_str = recommend(
            batch_size=3,
            searchspace_json=SEARCHSPACE_JSON,
            objective_json=OBJECTIVE_JSON,
            measurements_json=measurements,
        )
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_recommend_with_measurements_base64(self):
        """Recommendation with measurements in base64 format."""
        df = pd.DataFrame({"x1": [2.0, 4.0], "x2": [10.0, 30.0], "y": [0.3, 0.9]})
        measurements = json.dumps(converter.unstructure(df))
        result_str = recommend(
            batch_size=2,
            searchspace_json=SEARCHSPACE_JSON,
            objective_json=OBJECTIVE_JSON,
            measurements_json=measurements,
        )
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_recommend_output_base64(self):
        """Output in base64 format should be a base64-encoded string."""
        result_str = recommend(
            batch_size=1,
            searchspace_json=SEARCHSPACE_JSON,
            objective_json=OBJECTIVE_JSON,
            output_format="base64",
        )
        # base64 output is a JSON string containing a base64-encoded pickle
        result = json.loads(result_str)
        assert isinstance(result, str)
        # Should be deserializable back to a DataFrame
        df = converter.structure(result, pd.DataFrame)
        assert len(df) == 1

    def test_recommend_with_custom_recommender(self):
        """Custom recommender (RandomRecommender) should work."""
        rec_json = json.dumps({"type": "RandomRecommender"})
        result_str = recommend(
            batch_size=2,
            searchspace_json=SEARCHSPACE_JSON,
            objective_json=OBJECTIVE_JSON,
            recommender_json=rec_json,
        )
        result = json.loads(result_str)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_recommend_invalid_searchspace(self):
        """Invalid searchspace should return an error."""
        result_str = recommend(
            batch_size=1,
            searchspace_json='{"type": "SearchSpace", "invalid": true}',
            objective_json=OBJECTIVE_JSON,
        )
        result = json.loads(result_str)
        assert "error" in result
