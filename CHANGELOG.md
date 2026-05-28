# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-26

### Added
- REG (Relational Entity Graph) importance tracking
- Degree centrality + hit frequency based entity importance
- Automatic entity extraction to graph on events
- TTL-based node expiration (configurable, default 7 days)
- entity_pull_with_reg() for importance-ranked entity retrieval
- Multi-agent graph support (agents share entity graph)
- Proactive high-mass surface injection (REG injection) to auto-inject important nodes into active context

### Changed
- context_assembler now uses REG importance by default
- Entity retrieval now ranks by importance, not just recency
- .get_context() now packs important entities first

### Performance
- Added database indexes for graph queries
- TTL expiration runs on schedule (no impact on event writes)
- Background entity extraction (no impact on POST /events response time)

### Known Limitations
- Multi-agent entity ownership not yet implemented (coming in v0.3.0)
- Sentiment-based importance scoring deferred (GravityField research)

## [0.1.0] - 2026-05-24

### Added
- Append-only event log for source of truth
- Entity registry with conflict detection
- Automatic snapshot generation via LLM
- Semantic retrieval over compressed memory
- Write-back gate for validation
- Support for both SQLite and PostgreSQL

### Known Limitations
- No multi-agent coordination yet (coming in 0.2.0)
- Native pgvector acceleration requires PostgreSQL; falls back to in-memory cosine similarity on other databases
- Requires Python 3.10+
