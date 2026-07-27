# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
