# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-07-28
### Added
- `concepts` resource and tools (`list_concepts`, `get_concept`) exposing
  BayBE's concept explanation pages for the installed version.
- User-provided recipes: `.md` files in a recipes directory (`--recipes-dir`,
  default `recipes/`) are merged into the recipes index and served alongside the
  documentation-derived ones.
- Agent workflow guidance published as server instructions and reinforced in the
  `recommend` tool.

### Changed
- Renamed the `examples` resource and tools to `recipes`
  (`baybe://recipes`, `list_recipes`, `get_recipe`).
- The serialization guide is now served as a concept
  (`get_concept("serialization")`); the dedicated guide tool was removed.

## [0.4.0] - 2026-07-28
### Added
- `parameter_importance` tool: stateless SHAP-based parameter importance per
  target. Fits a surrogate on the provided measurements and returns the mean
  absolute SHAP value of each parameter, one row per parameter with a
  `<target>_importance` column per target.
- `parameter_importance` supports an optional surrogate config (default
  `GaussianProcessSurrogate`), the SHAP explainer choice (default
  `KernelExplainer`), and explaining the computational representation.
- The `baybe` dependency now includes the `chem` and `insights` extras
  (substance parameters and SHAP insights).

## [0.3.0] - 2026-07-28
### Added
- `predict` tool: stateless posterior statistics (predictions and uncertainty)
  for candidate points. Fits a surrogate on the provided measurements and
  returns the requested statistics.
- `predict` supports `mean`, `std`, `var`, `mode`, and quantile statistics, an
  optional surrogate config (default `GaussianProcessSurrogate`), and `records`
  or `base64` output.
- `predict` supports multi-target objectives (`ParetoObjective`,
  `DesirabilityObjective` without pre-transformation) by replicating a
  single-output surrogate per target, yielding per-target output columns.

## [0.2.0] - 2026-07-27
### Added
- Config-knowledge tools (`list_types`, `get_schema`, `get_serialization_guide`,
  `list_examples`, `get_example`, `get_docs_links`) that expose the same content
  as the resources, for MCP clients that surface only tools (e.g. OpenCode).
  Tools and resources share the same payload helpers, so they cannot drift.
- MCP resources exposing BayBE config knowledge, all derived from the installed
  BayBE version:
  - `baybe://types`: all serializable types grouped by family, each with its
    schema URI
  - `baybe://schema/{type}`: attribute fields, alternative `from_*` constructors,
    and docstring; nested objects carry a `$ref` to their schema
  - `baybe://guide/serialization`: the serialization guide fetched for the
    installed version, with an offline link fallback
  - `baybe://examples` and `baybe://examples/{topic}/{file}`: index and raw
    content of BayBE's example scenarios
  - `baybe://docs`: version-matched documentation links
- Version detection (`baybe.__version__`) driving all version-derived content,
  with a `latest` fallback when the version cannot be detected
- Portable, version-guarded resource cache with a manifest keyed by BayBE
  version and resource format version
- Standalone `build` command and cache-aware `run` startup that share a single
  `build_resources` routine; `run` flags `--cache-dir`, `--rebuild-resources`,
  and `--use-cache`

### Fixed
- `validate` and `recommend` now accept configs and measurements as native JSON
  objects/arrays in addition to JSON strings, so MCP clients that
  auto-deserialize valid-JSON arguments (e.g. OpenCode) work correctly. Search
  space and objective are structured via the converter, which also fixes
  `from_product` search spaces that were previously rejected.

## [0.1.0]
### Added
- `validate` tool: validates JSON configs for any serializable BayBE object
- `recommend` tool: stateless Bayesian optimization recommendations without a
  Campaign
- stdio and streamable-HTTP transports selectable via the CLI
- Pinned BayBE to `0.15.0`
