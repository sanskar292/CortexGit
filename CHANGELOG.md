# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-05-25

### Added
- Append-only event log for source of truth
- Entity registry with conflict detection
- Automatic snapshot generation via LLM
- Semantic retrieval over compressed memory
- Write-back gate for validation
- Support for both SQLite and PostgreSQL

### Known Limitations
- No multi-agent coordination yet (coming in 0.2.0)
- Vector search only works with pgvector, not all databases
- Requires Python 3.10+
