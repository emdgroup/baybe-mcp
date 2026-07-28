# baybe-mcp

MCP server that exposes [BayBE](https://github.com/emdgroup/baybe)'s Bayesian
optimization to AI agents. It offers stateless tools for recommendations,
predictions, and parameter importance, plus resources that teach an agent how to
build valid BayBE configurations for the installed BayBE version.

## Design Principles

- **Stateless.** No Campaign or server-side state is kept. Every tool call
  carries its full context (search space, objective, measurements).
- **Version-derived.** Types and schemata are introspected from the installed
  `baybe` package; concepts and recipes are fetched from the BayBE repository at
  the matching version tag. Nothing is hardcoded or shipped prebuilt.
- **Tool/resource parity.** All resource content is also exposed as tools (some
  clients surface only tools), and both delegate to the same code so they cannot
  drift.
- **Offline-capable.** Fetched content is stored in a version-guarded cache that
  can be built once and copied to an offline machine.
- **Guided workflow.** The server publishes an agent workflow: study concepts
  and recipes, learn serialization, study schemata, `validate`, then act.
- **Custom recipes.** Users can add their own `.md` recipes that are merged
  alongside the documentation-derived ones.

## Local Install

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/emdgroup/baybe-mcp.git
cd baybe-mcp
uv sync
```

BayBE is required at `baybe[chem,insights]>=0.15` (the `chem` and `insights`
extras enable substance parameters and SHAP-based insights). `uv.lock` pins the
exact resolved dependency set.

### OpenCode

Add to `~/.config/opencode/opencode.json` (or a project-level `opencode.json`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "baybe": {
      "type": "local",
      "command": ["uv", "run", "--directory", "/absolute/path/to/baybe-mcp", "python", "-m", "baybe_mcp.server", "run"],
      "enabled": true,
      "timeout": 600000
    }
  }
}
```

The timeout is set to 10 minutes. It covers the slow initial import of BayBE and
PyTorch, but recommendation calls themselves can take much longer, so you may
need to increase this value significantly.

### Claude Desktop

Add to `claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "baybe": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/baybe-mcp", "python", "-m", "baybe_mcp.server", "run"]
    }
  }
}
```

### Claude Code

Register the server with the CLI (use `--scope user` to make it available across
all projects instead of just the current one):

```bash
claude mcp add baybe -- uv run --directory /absolute/path/to/baybe-mcp python -m baybe_mcp.server run
```

Recommendation calls can take a long time. Raise Claude Code's per-tool timeout
via the `MCP_TOOL_TIMEOUT` environment variable (milliseconds), e.g.
`MCP_TOOL_TIMEOUT=600000 claude` for 10 minutes, and increase it significantly
if needed.

## Remote Install

Run the server as a persistent process; clients connect by URL:

```bash
uv run python -m baybe_mcp.server run --transport streamable-http
```

This serves the MCP endpoint at `http://127.0.0.1:8000/mcp`. Use `--host` /
`--port` to change the bind address. In production, add TLS and authentication
and use the public URL. A running HTTP server must be restarted to pick up code
changes.

Point clients at the URL:

```jsonc
// OpenCode
{ "mcp": { "baybe": { "type": "remote", "url": "http://127.0.0.1:8000/mcp", "enabled": true } } }
```

```jsonc
// Claude Desktop
{ "mcpServers": { "baybe": { "url": "http://127.0.0.1:8000/mcp" } } }
```

```bash
# Claude Code
claude mcp add --transport http baybe http://127.0.0.1:8000/mcp
```

## Resources, Caching, Custom Recipes

The server exposes resources (types, schemata, concepts, recipes, doc links)
derived from the installed BayBE version. Fetched content is stored in a cache
so the server starts fast and can run offline:

```bash
uv run python -m baybe_mcp.server build          # build the cache and exit
uv run python -m baybe_mcp.server run            # auto-builds if missing/stale
```

Building writes `.baybe_mcp_cache/` in the current directory (override with
`--cache-dir`). The cache is guarded by the BayBE version, a resource format
version, and a hash of the recipes directory, so it is rebuilt automatically
when any of these change. Because it is portable, it can be built on a networked
machine and copied to an offline one:

```bash
# online machine:
uv run python -m baybe_mcp.server build --cache-dir /path/to/cache
# copy the cache folder to the offline machine, then:
uv run python -m baybe_mcp.server run --use-cache --cache-dir /path/to/cache
```

**Custom recipes.** Drop your own `.md` files into a recipes directory
(default `recipes/`, override with `--recipes-dir`). Files in subfolders use the
subfolder name as their topic; top-level files are grouped under
`Custom_Recipes`. They appear in `list_recipes` / `baybe://recipes` tagged with
`source: user` and are baked into the cache at build time.

## Tools & Resources

**Action tools** (all stateless; config/dataframe arguments accept a JSON
object/array or a JSON string):

| Tool | Purpose | Key parameters |
|------|---------|----------------|
| `validate` | Validate a config for any BayBE object | `json_config` |
| `recommend` | Bayesian optimization recommendations (central tool) | `batch_size`, `searchspace_json`, `objective_json`, `measurements_json?`, `recommender_json?`, `pending_experiments_json?`, `output_format?` |
| `predict` | Posterior statistics for candidates | `searchspace_json`, `objective_json`, `candidates_json`, `measurements_json`, `stats?`, `surrogate_json?`, `output_format?` |
| `parameter_importance` | SHAP-based parameter importance per target | `searchspace_json`, `objective_json`, `measurements_json`, `surrogate_json?`, `explainer?`, `use_comp_rep?`, `output_format?` |

**Knowledge tools** (version-derived), each mirrored by a resource:

| Tool | Resource | Purpose |
|------|----------|---------|
| `list_types` | `baybe://types` | Serializable types grouped by family, with schema references |
| `get_schema` | `baybe://schema/{type}` | Fields, `from_*` constructors, docstring; nested `$ref`s |
| `list_concepts` | `baybe://concepts` | Index of concept explanation pages |
| `get_concept` | `baybe://concepts/{name}` | A concept page (e.g. `serialization`, `transfer_learning`) |
| `list_recipes` | `baybe://recipes` | Worked scenarios by topic (incl. custom recipes) |
| `get_recipe` | `baybe://recipes/{topic}/{file}` | A single recipe file |
| `get_docs_links` | `baybe://docs` | Version-matched documentation links |

**DataFrame formats** (measurements, candidates, pending experiments):

- **records**: `[{"x1": 1.0, "x2": 10.0, "y": 0.5}, ...]`
- **base64**: BayBE's native base64-encoded pickle string
- **constructor dict**: `{"constructor": "from_records", "data": [...]}`

Output is `"records"` (default) or `"base64"`.

## Tests

```bash
uv sync --extra test
uv run pytest
```
