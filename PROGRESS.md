## Last updated: 2026-05-26

## Phase: 3

## What is built and tested:
### Database & Schema Layer
- Defined standard `EventType` enum and built the four PostgreSQL/SQLAlchemy declarative models (`EventLog`, `EntityRegistry`, `SnapshotStore`, `ConflictLog`) in [models.py](file:///f:/aggin/CortexGit/db/models.py).
- Configured and initialized **Alembic** migrations in [alembic.ini](file:///f:/aggin/CortexGit/alembic.ini) and [env.py](file:///f:/aggin/CortexGit/db/migrations/env.py) to dynamically read connection settings from the `.env` file.
- Enforced append-only safety triggers on `event_log` updates/deletes and immutability triggers on `snapshot_store` updates at the PostgreSQL database level.
- Built a comprehensive database session dependency manager in [database.py](file:///f:/aggin/CortexGit/db/database.py) using asynchronous engines.
- Built a comprehensive programmatic integration test suite in [test_db_migration.py](file:///f:/aggin/CortexGit/tests/test_db_migration.py) verifying that tables exist, column schemas are correct, and database-level safety triggers raise SQL errors on illegal actions. All tests pass successfully.

### Event Logger Component
- Implemented the core `EventLogger` logic in [event_logger.py](file:///f:/aggin/CortexGit/core/event_logger.py) with event enum validation and clean write commits.
- Created Pydantic request/response validation schemas and implemented the public `POST /events` route in [events.py](file:///f:/aggin/CortexGit/api/routes/events.py), successfully registering it in [main.py](file:///f:/aggin/CortexGit/api/main.py).
- Built a complete integration test suite in [test_event_logger.py](file:///f:/aggin/CortexGit/tests/test_event_logger.py) that covers valid writes, 422 rejections for invalid event types, separate UUID generations, and validation of append-only constraints from the application layer. All tests pass successfully.

### Entity Registry & Conflict Detector Component
- Implemented `ConflictDetector` in [conflict_detector.py](file:///f:/aggin/CortexGit/core/conflict_detector.py) checking key collision and logging conflicts to the `conflict_log` database store.
- Implemented `EntityRegistryHandler` in [entity_registry.py](file:///f:/aggin/CortexGit/core/entity_registry.py) supporting clean and idempotent writes with strict event correlation.
- Created public `POST /entities` endpoint in [entities.py](file:///f:/aggin/CortexGit/api/routes/entities.py) with custom JSONResponse error styling to match required architecture formats, registering it in [main.py](file:///f:/aggin/CortexGit/api/main.py).
- Wrote robust integration tests in [test_entity_registry.py](file:///f:/aggin/CortexGit/tests/test_entity_registry.py) checking clean writes, idempotent same key/value writes, HTTP 409 conflict checks, correct database ConflictLog writes, and EntityRegistry mutation safety. All tests pass successfully.

### Write-Back Gate Component
- Implemented the standalone `WriteBackGate` validation utility in [write_back_gate.py](file:///f:/aggin/CortexGit/core/write_back_gate.py), reading JSONSchema documents dynamically from [schemas/](file:///f:/aggin/CortexGit/schemas/) and enforcing strict validation checks using `jsonschema`.
- Written comprehensive unit tests in [test_write_back_gate.py](file:///f:/aggin/CortexGit/tests/test_write_back_gate.py) verifying success paths, missing fields, type errors, unexpected additional properties, minLength string constraints, and empty array valid actions. All tests pass successfully.

### Recency Filter Component
- Implemented core `RecencyFilter` in [recency_filter.py](file:///f:/aggin/CortexGit/core/recency_filter.py) which retrieves the last `k` events for a session in chronological order (oldest to newest).
- Wrote integration tests in [test_recency_filter.py](file:///f:/aggin/CortexGit/tests/test_recency_filter.py) verifying that last `k` events are retrieved in order, default `k` is configurable per call, fewer than `k` events are returned if database contains fewer items, and empty lists are returned for unknown sessions.

### Vector Index & Semantic Recall Component (Phase 2)
- Implemented `embed_text()` utility in [embeddings.py](file:///f:/aggin/CortexGit/retrieval/embeddings.py) utilizing OpenAI's `text-embedding-3-small` model to generate text embedding vectors.
- Configured and created a new Alembic migration in [bfc3db3b55cf_enable_pgvector.py](file:///f:/aggin/CortexGit/db/migrations/versions/bfc3db3b55cf_enable_pgvector.py) that enables the `vector` extension and alters the `snapshot_store` table's `embedding` column type to `vector(1536)` cleanly.
- Built an integration test suite in [test_embeddings.py](file:///f:/aggin/CortexGit/tests/test_embeddings.py) marked with `@pytest.mark.integration` to verify real OpenAI embeddings generation. All tests pass successfully.
- Implemented `semantic_recall()` core function in [semantic_recall.py](file:///f:/aggin/CortexGit/retrieval/semantic_recall.py) running database-side `cosine_distance` queries on the `snapshot_store` table.
- Built an offline-safe unit and integration test suite in [test_semantic_recall.py](file:///f:/aggin/CortexGit/tests/test_semantic_recall.py) mocking embeddings and asserting empty-database results, top_n limiting, and correct similarity sorting order. All tests pass successfully.

### Entity Pull Component (Phase 2)
- Implemented `entity_pull()` core function in [entity_pull.py](file:///f:/aggin/CortexGit/core/entity_pull.py) extracting alphanumeric word tokens from the goal string and executing case-insensitive substring matches against entity registry keys.
- Built a comprehensive integration test suite in [test_entity_pull.py](file:///f:/aggin/CortexGit/tests/test_entity_pull.py) validating single-token matching, multiple-token matching, case insensitivity, multiple matches, and empty results. All tests pass successfully.

### Context Assembler Component (Phase 2)
- Implemented `assemble()` core function in [context_assembler.py](file:///f:/aggin/CortexGit/core/context_assembler.py) integrating all Phase 2 retrieval modules to construct packed context datasets.
- Implemented deterministic JSON-based token estimation and sequential priority-based context packing (Conflicts -> Recent Events -> Snapshots -> Entities).
- Built a comprehensive integration test suite in [test_context_assembler.py](file:///f:/aggin/CortexGit/tests/test_context_assembler.py) validating key outputs, zero-budget empty results, strict budget token enforcement, and correct priority execution. All tests pass successfully.

### Context Route Component (Phase 2)
- Created public `GET /context` endpoint in [context.py](file:///f:/aggin/CortexGit/api/routes/context.py) utilizing standard FastAPI query params validation and explicit whitespace rejection. Registered it in [main.py](file:///f:/aggin/CortexGit/api/main.py).
- Built a comprehensive integration test suite in [test_context_route.py](file:///f:/aggin/CortexGit/tests/test_context_route.py) validating key outputs, whitespace string rejection, zero and negative budget rejection, and empty-stores gracefully returning empty lists. All tests pass successfully.

### Snapshot Trigger Component (Phase 3)
- Extended database schema by adding `session_id` column to `SnapshotStore` model, establishing absolute multi-session isolation of snapshots.
- Generated and applied Alembic migration `0425218591b2_add_session_id_to_snapshot` to add the indexed column to the local PostgreSQL database.
- Implemented the `should_snapshot` trigger function and the `SnapshotTrigger` class in [snapshot_trigger.py](file:///f:/aggin/CortexGit/llm/snapshot_trigger.py).
- Built a comprehensive test suite in [test_snapshot_trigger.py](file:///f:/aggin/CortexGit/tests/test_snapshot_trigger.py) validating count threshold logic, configurable limits, and empty-snapshot boundaries. All tests pass successfully.

### Summarizer LLM Component (Phase 3)
- Implemented the `summarize` function and `Summarizer` class in [summarizer.py](file:///f:/aggin/CortexGit/llm/summarizer.py) utilizing the Anthropic SDK with model `claude-sonnet-4-20250514`.
- Uses the exact memory summarizer system prompt defined in `ARCHITECTURE.md` with no modifications or retries.
- Enforces strict data safety by routing all LLM responses through the Write-Back Gate (`WriteBackGate.validate()` with `schema_name="snapshot"`) and bubbles up validation errors immediately.
- Wrote robust tests in [test_summarizer.py](file:///f:/aggin/CortexGit/tests/test_summarizer.py) marked with `@pytest.mark.integration` to verify schema validation, missing fields, and unexpected properties using mocked API responses. All tests pass successfully.

### Entity Extractor LLM Component (Phase 3)
- Implemented `extract_entities` function and `EntityExtractor` class in [entity_extractor.py](file:///f:/aggin/CortexGit/llm/entity_extractor.py) utilizing the Anthropic SDK with model `claude-sonnet-4-20250514`.
- Uses the exact entity extractor system prompt defined in `ARCHITECTURE.md` with no modifications.
- Enforces validation through the Write-Back Gate (`WriteBackGate.validate()` with `schema_name="entity_extraction"`) and raises `ValidationError` immediately on failure.
- Wired up the asynchronous background task `run_entity_extraction_pipeline` in [events.py](file:///f:/aggin/CortexGit/api/routes/events.py) to run after every successful event write via FastAPI `BackgroundTasks`.
- Background task loops through extracted updates, checks collisions via `ConflictDetector`, logs conflicts to `ConflictLog` on collision, and performs clean or idempotent updates in `EntityRegistry` using `EntityRegistryHandler`.
- Handles `ValidationError` and other API exceptions gracefully within the background pipeline to prevent request/process crashes.
- Built a comprehensive test suite in [test_entity_extractor.py](file:///f:/aggin/CortexGit/tests/test_entity_extractor.py) verifying valid writes, partial validation failures, invalid key patterns, empty updates handling, and collision logging. All 6 tests pass successfully.

### SDK Packaging & Distribution (Final Release)
- Restructured all modules with appropriate `__init__.py` files to ensure seamless auto-discovery via `setuptools`.
- Configured `.github/workflows/tests.yml` to run automated testing, linting (`ruff`), and code formatting (`black`) using a PostgreSQL service container on every commit or pull request.
- Verified local installation in an isolated virtual environment (`test_env`) using `python -m build` and `pip install`.
- Published the official release version `v0.1.0` to PyPI, enabling global installation via `pip install cortexgit`.
- Created comprehensive developer and user documentation including `docs/GETTING_STARTED.md`, `docs/API_REFERENCE.md`, `CONTRIBUTING.md`, `RELEASE_CHECKLIST.md`, and streamlined `.env.example`.


## Decisions made:
- Integrated an automatic fallback for the `embedding` column: if the `vector` extension is not registered on the PostgreSQL server, the column gracefully maps to `postgresql.ARRAY(Float)`.
- Overrode FastAPI's `get_db` dependency programmatically in test fixtures to inject custom asynchronous test sessions linked to isolated test databases.
- Configured test engine connection pools with `NullPool` in async test modules to guarantee immediate teardown and prevent locks or "closed event loop" exceptions in teardown phases.
- Enforced strict event correlation on `POST /entities`: require `event_id` explicitly in the POST payload and completely removed the synthetic default event fallback mechanism to guarantee absolute database consistency and causal ordering.
- Implemented a completely standalone `WriteBackGate` with zero external dependencies to decouple data validation from system database or API router setups.
- Implemented concurrency and safety fixes (Fix 1 & Fix 2) on `POST /entities`: added row locking using `.with_for_update()` in conflict check/write, and robust error recovery using `db.rollback()` to prevent unhandled 500 errors on concurrent duplicate key writes, returning `409 Conflict` and properly logging the conflict in `ConflictLog`.
- Designed standard database `created_at.desc()` query sorting for `RecencyFilter` and reversed the resulting collection in the application layer to return events in precise oldest-to-newest chronological order.
- Implemented defensive dynamic check of `pg_available_extensions` in the Alembic migration to enable the pgvector extension and migrate columns safely without causing transaction aborts on systems lacking local pgvector extension support.
- Registered the custom `@pytest.mark.integration` marker in `pytest.ini` at the root directory to manage integration tests cleanly and avoid test output warnings.
- Implemented defensive dynamic check of `HAS_PGVECTOR` in `semantic_recall()` to fall back to an in-memory python-based cosine similarity computation when running on environments lacking the pgvector database extension (such as local test runs), guaranteeing seamless execution and verification.
- Standardized `re.findall(r'[a-zA-Z0-9]+', goal)` in `entity_pull()` to split goal queries on arbitrary spaces and punctuation, cleanly isolating search tokens.
- Structured integration tests in `test_entity_pull.py` to insert valid helper events first, satisfying database foreign key requirements before registering test entities.
- Standardized alphabetized entity key ordering and primary-key sorting in the Context Assembler's packing routines to guarantee absolute determinism in generated outputs.
- Standardized FastAPI's `Query` validation combined with explicit `.strip()` verification on endpoints to cleanly capture empty and whitespace-only query inputs before reaching database components.
- Added an indexed `session_id` column to `SnapshotStore` to allow proper isolation of memory snapshots between concurrent agent sessions and to enable correct snapshot triggering.
- Implemented robust `event_range` parsing logic in `should_snapshot` to seamlessly extract inclusive upper bounds from PostgreSQL Range objects, lists, or tuples, guaranteeing maximum runtime durability.
- Extended background task queue in `POST /events` to execute both the Entity Extraction pipeline and the Snapshot Trigger pipeline concurrently.
- Implemented transactional collision-handling and rollbacks on duplicate concurrent keys inside the background pipeline, aligning perfectly with the core `POST /entities` controller.
- Upgraded test suite mock frameworks (`test_snapshot_write.py`) to process lists of concurrent background tasks, correcting mock-collision behavior.
- Implemented and verified the full end-to-end integration test (`tests/test_e2e_full_pipeline.py`) posting 50 realistic events, draining background tasks (triggering both entity extraction and snapshotting), validating strict token budget packing in `GET /context`, writing a 51st event triggering a conflict, and verifying the conflict shows up in `/context`.
- Added critical dependency requirements (`python-dotenv`, `asyncpg`, and `aiosqlite`) directly to `setup.py` and `pyproject.toml` to prevent missing package imports upon library installation.
- Standardized package versioning string `"0.1.0"` across exactly 3 files: `setup.py`, `pyproject.toml`, and `src/cortexgit/__init__.py`.
- Formulated a standard MIT License in the project root under copyright year 2024.
- Added Multi-Provider LLM and Embedding Support layer in `src/cortexgit/llm_providers/` folder:
  - Created `LLMProvider` and `EmbeddingProvider` abstract base classes with unified exceptions (`LLMError`, `EmbeddingError`).
  - Implemented four concrete provider modules (`AnthropicProvider`, `OpenAIProvider`, `OpenRouterProvider`, `OllamaProvider`) supporting both LLM completions and embeddings generation.
  - Implemented `provider_factory.py` creating instances based on names, reading from environment variables or custom kwargs.
  - Integrated providers into `CortexGit` client in `src/cortexgit/core/memory.py` and documented all configuration options in `.env.example`.
  - Wired LLM provider calls in `summarizer.py` and `entity_extractor.py` to use `self.llm_provider`, maintaining compatibility with legacy mocks using dynamic fallback checks.
  - Wired Embedding provider calls in `embeddings.py`, `semantic_recall.py`, and `context_assembler.py` to use `self.embedding_provider` cleanly.
  - Built comprehensive unit/mock and integration test suites validating all providers, factory creation, and runtime provider wiring.
  - Created `docs/PROVIDERS.md` cost and quickstart documentation and updated `README.md`.

*Total System Tests: 92 collected (82 passed offline, 10 marked integration).*

## Known issues:
- None.

## Next session starts with:
- Implementing multi-agent coordination capabilities (v0.2.0) or adding custom telemetry and logging to LLM provider classes.


