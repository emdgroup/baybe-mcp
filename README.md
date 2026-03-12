# baybe-mcp

MCP server that exposes [BayBE](https://github.com/emdgroup/baybe)'s Bayesian optimization as tools for AI agents.

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running the server

```bash
uv run python -m baybe_mcp.server
```

This starts the MCP server on stdio (the default MCP transport).

## Tools

### `validate`

Validates a JSON configuration for any BayBE object. Useful for agents to check their configs before passing them to `recommend`.

**Input:**
- `json_config` (string): JSON string with a `"type"` field identifying the concrete BayBE class.

**Returns:** JSON with `"valid"` (bool) and `"message"` (string).

**Example:**
```json
{
  "type": "NumericalDiscreteParameter",
  "name": "temperature",
  "values": [100.0, 150.0, 200.0]
}
```

### `recommend`

Performs a stateless Bayesian optimization recommendation. No Campaign or server-side state -- all context is passed per call.

**Inputs:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `batch_size` | int | yes | Number of experiments to recommend |
| `searchspace_json` | string | yes | JSON-serialized [SearchSpace](https://emdgroup.github.io/baybe/stable/userguide/searchspace.html) |
| `objective_json` | string | yes | JSON-serialized [Objective](https://emdgroup.github.io/baybe/stable/userguide/objectives.html) |
| `measurements_json` | string | no | Past measurements as a DataFrame (see formats below) |
| `recommender_json` | string | no | Recommender config. Default: `TwoPhaseMetaRecommender` |
| `pending_experiments_json` | string | no | Pending experiments DataFrame |
| `output_format` | string | no | `"records"` (default) or `"base64"` |

**Returns:** JSON-serialized DataFrame of recommended experiments.

**Example searchspace:**
```json
{
  "type": "SearchSpace",
  "constructor": "from_product",
  "parameters": [
    {"type": "NumericalDiscreteParameter", "name": "x1", "values": [1.0, 2.0, 3.0]},
    {"type": "NumericalDiscreteParameter", "name": "x2", "values": [10.0, 20.0, 30.0]}
  ]
}
```

**Example objective:**
```json
{
  "type": "SingleTargetObjective",
  "target": {
    "type": "NumericalTarget",
    "name": "y",
    "transformation": {"type": "IdentityTransformation"},
    "minimize": false
  }
}
```

## DataFrame Formats

Measurements and pending experiments accept three formats:

- **records** (agent-friendly): `[{"x1": 1.0, "x2": 10.0, "y": 0.5}, ...]`
- **base64**: BayBE's native base64-encoded pickle string (from `converter.unstructure(df)`)
- **constructor dict**: `{"constructor": "from_records", "data": [...]}`

Output can be `"records"` (list of dicts) or `"base64"`.

## Tests

```bash
uv sync --extra test
uv run pytest
```

## References

- [BayBE documentation](https://emdgroup.github.io/baybe/stable/)
- [BayBE serialization guide](https://emdgroup.github.io/baybe/stable/userguide/serialization.html)
- [Stateless recommend call](https://emdgroup.github.io/baybe/stable/userguide/getting_recommendations.html#the-recommend-call)
