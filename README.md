# baybe-mcp

MCP server that exposes [BayBE](https://github.com/emdgroup/baybe)'s Bayesian optimization as tools for AI agents.

## Setup

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

BayBE is pinned to an exact version (`baybe==0.15.0`) in `pyproject.toml`, and the fully resolved dependency set is committed in `uv.lock`.

### Checking installed versions

Any user can inspect which versions are actually installed:

```bash
uv run python -c "import baybe; print(baybe.__version__)"  # BayBE version
uv pip show baybe mcp                                      # specific packages
uv pip list                                                # all installed packages
```

`uv.lock` is the source of truth for the exact resolved versions of all dependencies.

## Building resources

The server exposes resources (types, schemas, a serialization guide, examples)
that are derived from the installed BayBE version. They are stored in a cache
directory so the server starts fast and can run offline.

```bash
uv run python -m baybe_mcp.server build
```

This writes `.baybe_mcp_cache/` in the current directory. Building is optional:
the server auto-builds on first start when the cache is missing/stale and the
network is reachable. Use `--cache-dir` to change the location.

Because everything is derived from the installed BayBE version, you only ever
need to choose the BayBE version. The cache is version-guarded, so it can be
built on one machine and copied to another (e.g. one without internet):

```bash
# machine with internet:
uv run python -m baybe_mcp.server build --cache-dir /path/to/cache
# copy the cache folder to the offline machine, then:
uv run python -m baybe_mcp.server run --use-cache --cache-dir /path/to/cache
```

## Running the server

The server supports two transports. Run `run --help` to see all options
(`--transport`, `--host`, `--port`, `--log-level`, `--cache-dir`,
`--rebuild-resources`, `--use-cache`).

### stdio (local)

The MCP client spawns the server as a subprocess and talks over stdin/stdout:

```bash
uv run python -m baybe_mcp.server run
```

### HTTP (remote)

The server runs as a persistent process and clients connect by URL. This is the
mode used for hosting the server remotely:

```bash
uv run python -m baybe_mcp.server run --transport streamable-http
```

This serves the MCP endpoint at `http://127.0.0.1:8000/mcp`. Use `--host` /
`--port` to change the bind address, and `--log-level DEBUG` to see request and
tool traffic while debugging.

Notes:
- In production, only the URL changes (e.g. `https://your-host.example.com/mcp`);
  you would additionally add TLS and authentication.
- Binding beyond localhost triggers the SDK's DNS-rebinding protection, which
  rejects unknown hosts by default.
- A running HTTP server must be **restarted** to pick up code changes.

## Connecting from OpenCode

Add the server to your OpenCode config (global `~/.config/opencode/opencode.json`
or a project-level `opencode.json`).

### stdio

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "baybe": {
      "type": "local",
      "command": ["uv", "run", "--directory", "/absolute/path/to/baybe-mcp", "python", "-m", "baybe_mcp.server", "run"],
      "enabled": true,
      "timeout": 30000
    }
  }
}
```

The raised `timeout` accounts for the slow initial import of BayBE and PyTorch,
which exceeds OpenCode's 5000ms default.

### HTTP (remote)

Start the server in HTTP mode first, then point OpenCode at its URL:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "baybe": {
      "type": "remote",
      "url": "http://127.0.0.1:8000/mcp",
      "enabled": true
    }
  }
}
```

Once connected, the `baybe` tools are available to the agent alongside its
built-in tools.

## Tools

**Recommended workflow:** discover types with `list_types`, read a type's
`get_schema` (consult `get_serialization_guide` and `list_examples` /
`get_example` for patterns), build the config, confirm it with `validate`, then
call `recommend`.

### `validate`

Validates a JSON configuration for any BayBE object. Useful for agents to check their configs before passing them to `recommend`.

**Input:**
- `json_config` (JSON object or string): a config with a `"type"` field identifying the concrete BayBE class.

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

Config and measurement inputs accept either a JSON object/array or a JSON
string; both are handled transparently (some MCP clients auto-deserialize valid
JSON arguments).

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

## Config-knowledge tools

These tools help agents build valid configs. All content is derived from the
installed BayBE version.

| Tool | Description |
|------|-------------|
| `list_types` | All serializable BayBE types, grouped by family, each with its schema reference. |
| `get_schema` | Fields (name, type, default, required), alternative `from_*` constructors, and docstring for a type. Nested objects carry a `$ref` to their own schema. |
| `get_serialization_guide` | The BayBE serialization guide for the installed version (fetched; links out when offline). |
| `list_examples` | Index of BayBE example scenarios (topics and files). |
| `get_example` | Raw content of a single example scenario file. |
| `get_docs_links` | Version-matched links to the BayBE documentation. |

## Resources

The same content is also exposed as MCP resources, for clients that surface
resources to the agent (some clients, such as OpenCode, expose only tools —
use the tools above with those).

| URI | Equivalent tool |
|-----|-----------------|
| `baybe://types` | `list_types` |
| `baybe://schema/{type}` | `get_schema` |
| `baybe://guide/serialization` | `get_serialization_guide` |
| `baybe://examples` | `list_examples` |
| `baybe://examples/{topic}/{file}` | `get_example` |
| `baybe://docs` | `get_docs_links` |

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
