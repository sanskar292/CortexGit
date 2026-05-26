"""
End-to-end integration test for the full CortexGit pipeline.

Requirements:
  - PostgreSQL running at localhost:5432 (password: password)
  - pgvector extension installed
  - Real ANTHROPIC_API_KEY in environment (not mock_key)
  - Real OPENAI_API_KEY in environment (not mock_key)

Run with:
  python -m pytest tests/test_e2e_full_pipeline.py -m integration -s -v

The test writes 50 realistic agent events to a single session, verifies entity extraction,
generates a snapshot on the 50th event, exercises get_context, and triggers conflicts.

No LLM calls are mocked — this validates the real prompts end-to-end.
"""

import asyncio
import json
import os
import uuid

import psycopg2
import pytest
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import cortexgit.db.database
from cortexgit import CortexGit
from cortexgit.core.memory import ConflictError
from cortexgit.db.models import Base, ConflictLog, EntityRegistry, SnapshotStore

# ──────────────────────────────────────────────────────────────────────────────
# Test database setup
# ──────────────────────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/cortexgit_e2e_test"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestingSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

# Redirect the module-level AsyncSessionLocal so background tasks use the test DB
cortexgit.db.database.AsyncSessionLocal = TestingSessionLocal


@pytest.fixture(autouse=True, scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module", autouse=True)
def create_test_db():
    """Create the e2e test database; tear it down after the module."""
    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_e2e_test;")
    cur.execute("CREATE DATABASE cortexgit_e2e_test;")
    cur.close()
    conn.close()

    yield

    asyncio.run(test_engine.dispose())

    conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("DROP DATABASE IF EXISTS cortexgit_e2e_test;")
    cur.close()
    conn.close()


@pytest.fixture(scope="module", autouse=True)
async def setup_db(create_test_db):
    """Enable pgvector and create all tables once for the module."""
    async with test_engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            pass  # pgvector may already exist or not be available; fallback handles it

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ──────────────────────────────────────────────────────────────────────────────
# Realistic agent event corpus
# ──────────────────────────────────────────────────────────────────────────────

AGENT_ID = "cortex-planner-v1"
SESSION_ID = f"e2e-session-{uuid.uuid4().hex[:8]}"

# 50 realistic events that tell a coherent story of an agent planning and
# executing a software migration from a monolith to microservices.
EVENTS = [
    # Session bootstrap
    ("system",  {"text": "Session initialised. Agent cortex-planner-v1 is starting a new planning cycle."}),
    ("user",    {"text": "We need to migrate the billing service from the monolith to its own microservice by end of Q3."}),
    ("agent",   {"text": "Understood. I'll begin by auditing the billing module's current dependencies and API surface."}),
    # Discovery phase
    ("action",  {"text": "Running dependency graph analysis on billing module.", "tool": "dep_graph", "args": {"module": "billing"}}),
    ("observation", {"text": "Found 14 internal dependencies: payments, users, subscriptions, audit_log, email, pdf_renderer, feature_flags, rate_limiter, config, metrics, cache, db_pool, auth, notifications."}),
    ("thought", {"text": "The high dependency count suggests significant coupling. I should prioritise identifying which dependencies can be extracted vs. which need to be replaced with API calls."}),
    ("action",  {"text": "Querying code ownership for billing module.", "tool": "git_blame_summary", "args": {"path": "src/billing"}}),
    ("observation", {"text": "Primary owners: alice@company.com (62%), bob@company.com (28%), carol@company.com (10%). Last significant change: 3 weeks ago. Alice is flagged as the primary technical reviewer for architecture decisions."}),
    ("action",  {"text": "Fetching open issues labelled billing.", "tool": "issue_tracker", "args": {"label": "billing", "state": "open"}}),
    # Planning phase
    ("observation", {"text": "27 open issues: 8 bugs, 14 feature requests, 5 tech-debt items. Critical bug: invoice generation occasionally produces duplicate line items (BILL-2041)."}),
    ("thought", {"text": "BILL-2041 must be fixed before migration. A bug carried across service boundaries becomes much harder to debug."}),
    ("agent",   {"text": "Setting project goal: migrate billing to microservice by 2026-09-30. Pre-migration blocker: resolve BILL-2041."}),
    ("user",    {"text": "Can you also check what the current test coverage is?"}),
    ("action",  {"text": "Running test coverage report for billing module.", "tool": "coverage_tool", "args": {"module": "billing"}}),
    ("observation", {"text": "Current test coverage: 41%. Industry recommendation for critical billing code: 85%+. Gap: 44 percentage points."}),
    ("thought", {"text": "41% coverage is dangerously low for financial code. We need a coverage improvement sprint before any migration work begins."}),
    ("agent",   {"text": "I'm adding a coverage improvement milestone. Target: 80% coverage on billing before migration begins."}),
    # Architecture design
    ("user",    {"text": "What deployment model are we targeting? Kubernetes or serverless?"}),
    ("thought", {"text": "Given the billing service's stateful nature and SLA requirements, Kubernetes is a better fit than serverless. Serverless cold starts could cause payment timeouts."}),
    ("agent",   {"text": "Recommending Kubernetes deployment. Billing has strict latency SLAs (p99 < 200ms) incompatible with serverless cold start profiles."}),
    ("action",  {"text": "Generating infrastructure cost estimate.", "tool": "cost_estimator", "args": {"service": "billing", "platform": "kubernetes", "replicas": 3}}),
    ("observation", {"text": "Estimated monthly cost: $840 for 3 replicas (2 vCPU, 4GB RAM each) plus $120 for managed PostgreSQL. Total: ~$960/month."}),
    ("agent",   {"text": "Cost estimate approved internally. Proceeding with Kubernetes architecture."}),
    # Data migration planning
    ("user",    {"text": "How do we handle the data migration? Billing data can't have any downtime."}),
    ("thought", {"text": "A dual-write strategy with a cutover window is the safest approach. We write to both the monolith DB and the new service DB during a transition period, then flip the read source once we verify consistency."}),
    ("agent",   {"text": "Proposing dual-write migration strategy with a 2-week validation window. No downtime expected. Rollback: revert read source to monolith."}),
    ("action",  {"text": "Checking current billing DB schema version.", "tool": "schema_inspector", "args": {"db": "billing"}}),
    ("observation", {"text": "Schema version: 47. Tables: invoices (2.1M rows), line_items (18.7M rows), payment_methods (890K rows), audit_events (45M rows). Largest table: audit_events."}),
    ("thought", {"text": "45M audit events is a significant migration risk. We should migrate audit_events asynchronously after the main service is live, using a background job."}),
    # Risk register
    ("agent",   {"text": "Creating risk register entry: audit_events table (45M rows) will migrate asynchronously post-cutover. Estimated migration time: 6 hours via streaming."}),
    ("action",  {"text": "Identifying API consumers of billing module.", "tool": "api_dependency_scanner", "args": {"module": "billing"}}),
    ("observation", {"text": "14 internal API consumers identified: order_service, cart_service, admin_portal, reporting_service, webhook_dispatcher, cron_jobs (3 distinct), mobile_api (iOS), mobile_api (Android), customer_portal, partner_api, data_pipeline, ml_feature_store."}),
    ("thought", {"text": "14 consumers means the migration must include a backward-compatible API shim. All consumers must be updated within the same sprint to avoid extended shim maintenance."}),
    ("agent",   {"text": "Adding migration sprint scope: update all 14 API consumers to point at new billing service endpoint. Shim will be maintained for 30 days max."}),
    # Timeline finalisation
    ("user",    {"text": "What's the final timeline?"}),
    ("agent",   {"text": "Proposed timeline: Sprint 1 (June): Fix BILL-2041, raise coverage to 80%. Sprint 2 (July): Build billing microservice, write tests. Sprint 3 (August): Dual-write, validate consistency. Sprint 4 (September): Cutover, update consumers, decommission monolith billing module."}),
    ("user",    {"text": "Looks good. Please lock the project goal and owner."}),
    ("agent",   {"text": "Locking: project.goal = migrate billing to microservice by 2026-09-30. project.owner = alice@company.com. project.status = in_progress."}),
    # Implementation begins
    ("action",  {"text": "Creating JIRA epic for billing microservice migration.", "tool": "jira", "args": {"type": "epic", "title": "Billing Microservice Migration", "assignee": "alice", "due": "2026-09-30"}}),
    ("observation", {"text": "Epic created: BILL-EPIC-001. 4 sub-stories automatically generated from sprint breakdown."}),
    ("action",  {"text": "Setting up new GitHub repository for billing microservice.", "tool": "github", "args": {"repo": "billing-service", "template": "python-fastapi-service", "team": "billing-team"}}),
    ("observation", {"text": "Repository created: github.com/company/billing-service. CI/CD pipeline configured with pytest, mypy, and Docker build stages."}),
    ("thought", {"text": "The repository is ready. Next step is to scaffold the domain model and begin porting the invoice generation logic, starting with the duplicate line item bug."}),
    ("action",  {"text": "Assigning BILL-2041 to alice with P0 priority.", "tool": "jira", "args": {"issue": "BILL-2041", "assignee": "alice", "priority": "P0", "sprint": "Sprint 1"}}),
    ("observation", {"text": "BILL-2041 assigned to alice, priority set to P0, added to Sprint 1."}),
    # Status update
    ("user",    {"text": "Any blockers I should know about?"}),
    ("thought", {"text": "The main risk remains coverage. If we can't hit 80% before July, the July build sprint will slip. I should flag this as a watch item."}),
    ("agent",   {"text": "Flagging: test coverage is the primary schedule risk. If coverage milestone is not met by June 30, recommend delaying the Sprint 2 start by one week. Secondary risk: 14 consumer updates in one sprint is aggressive. Recommending a feature freeze on all billing-dependent services during Sprint 4 cutover week."}),
    # Session wrap
    ("agent",   {"text": "Planning session complete. Summary: billing microservice migration locked for Q3. Owner: alice. Blocker: BILL-2041. Coverage target: 80%. Deployment: Kubernetes. Data strategy: dual-write. Timeline: 4 sprints ending 2026-09-30."}),
    ("system",  {"text": "Session checkpoint saved. Next session: review Sprint 1 progress and validate BILL-2041 fix."}),
]

assert len(EVENTS) == 50, f"Expected 50 events, got {len(EVENTS)}"


# ──────────────────────────────────────────────────────────────────────────────
# The test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.anyio
async def test_full_pipeline_50_events_snapshot_entities_context_conflict():
    """
    Full end-to-end pipeline test with real Anthropic and OpenAI API calls.

    Phases:
      A. Write 50 events using SDK log_event method.
      B. Assert snapshot was created.
      C. Assert entities were extracted.
      D. Call get_context and verify events, snapshot, entities, budget.
      E. Write a conflicting event that triggers a conflict.
      F. Assert get_context now includes the conflict.
    """
    memory = CortexGit(database_url=TEST_DATABASE_URL)

    # ──────────────────────────────────────────────────────────────────────
    # Phase A: Write 50 events
    # ──────────────────────────────────────────────────────────────────────
    last_event = None
    for i, (event_type, payload) in enumerate(EVENTS, start=1):
        last_event = await memory.log_event(
            session_id=SESSION_ID,
            agent_id=AGENT_ID,
            event_type=event_type,
            payload=payload,
        )
        assert last_event is not None
        assert last_event.event_id is not None

    # ──────────────────────────────────────────────────────────────────────
    # Phase B: Confirm a snapshot was created
    # ──────────────────────────────────────────────────────────────────────
    async with TestingSessionLocal() as session:
        snapshots = (
            await session.execute(
                select(SnapshotStore).where(SnapshotStore.session_id == SESSION_ID)
            )
        ).scalars().all()

    assert len(snapshots) >= 1, (
        "Expected at least one snapshot after 50 events, but snapshot store is empty. "
        "Check that ANTHROPIC_API_KEY and OPENAI_API_KEY are set to real values."
    )

    snapshot = snapshots[0]
    assert snapshot.session_id == SESSION_ID
    assert len(snapshot.summary) >= 10, "Snapshot summary is too short."
    assert isinstance(snapshot.entities_mentioned, list)
    assert snapshot.embedding is not None, "Snapshot embedding was not written."

    # ──────────────────────────────────────────────────────────────────────
    # Phase C: Confirm entities were extracted
    # ──────────────────────────────────────────────────────────────────────
    async with TestingSessionLocal() as session:
        entities = (await session.execute(select(EntityRegistry))).scalars().all()

    assert len(entities) >= 1, (
        "Expected at least one entity in the registry after 50 realistic events, "
        "but the registry is empty. Check ANTHROPIC_API_KEY."
    )

    # Build a key→value map for later phases
    entity_map: dict = {e.key: {"value": e.value, "event_id": e.event_id} for e in entities}
    print(f"\n[Phase C] Extracted entities ({len(entity_map)}): {list(entity_map.keys())}")

    # ──────────────────────────────────────────────────────────────────────
    # Phase D: get_context — verify response structure and token budget
    # ──────────────────────────────────────────────────────────────────────
    budget = 8000
    ctx = await memory.get_context(
        goal="billing microservice migration plan and project status",
        budget_tokens=budget,
        session_id=SESSION_ID,
    )

    assert "events" in ctx
    assert "snapshots" in ctx
    assert "entities" in ctx
    assert "conflicts" in ctx

    # At least some events returned (recency filter returns up to 20)
    assert len(ctx["events"]) >= 1, (
        "Expected at least one event in context response."
    )

    # At least one snapshot in context
    assert len(ctx["snapshots"]) >= 1, (
        "Expected at least one snapshot in context response."
    )

    # At least one entity in context
    assert len(ctx["entities"]) >= 1, (
        "Expected at least one entity in context response."
    )

    # Token budget must not be exceeded
    raw_json = json.dumps(ctx)
    estimated_tokens = len(raw_json) // 4
    assert estimated_tokens <= budget, (
        f"Context response exceeded budget: estimated {estimated_tokens} tokens > {budget}."
    )

    print(f"[Phase D] Context: {len(ctx['events'])} events, "
          f"{len(ctx['snapshots'])} snapshots, "
          f"{len(ctx['entities'])} entities, "
          f"~{estimated_tokens} estimated tokens (budget={budget}).")

    # ──────────────────────────────────────────────────────────────────────
    # Phase E: Write a 51st event that conflicts with an extracted entity
    # ──────────────────────────────────────────────────────────────────────
    conflict_key = "project.owner"
    if conflict_key not in entity_map:
        conflict_key = list(entity_map.keys())[0]

    original_value = entity_map[conflict_key]["value"]
    conflicting_value = "bob@company.com" if conflict_key == "project.owner" else "new_conflict_value"

    print(f"[Phase E] Writing 51st event to trigger conflict on key='{conflict_key}', "
          f"original='{original_value}', proposed='{conflicting_value}'")

    # Post the 51st event
    event_51 = await memory.log_event(
        session_id=SESSION_ID,
        agent_id=AGENT_ID,
        event_type="agent",
        payload={
            "text": f"Change of plans: {conflict_key} is now {conflicting_value}."
        },
    )

    # Direct write conflict check
    with pytest.raises(ConflictError):
        await memory.write_entity(
            key=conflict_key,
            value=conflicting_value,
            agent_id="conflict-agent",
            event_id=event_51.event_id,
        )

    # Let's verify that a conflict was indeed logged for this key in the database
    async with TestingSessionLocal() as session:
        conflicts = (
            await session.execute(
                select(ConflictLog).where(ConflictLog.key == conflict_key)
            )
        ).scalars().all()

    assert len(conflicts) >= 1

    # ──────────────────────────────────────────────────────────────────────
    # Phase F: get_context must now include the conflict
    # ──────────────────────────────────────────────────────────────────────
    ctx2 = await memory.get_context(
        goal="billing microservice migration plan and project status",
        budget_tokens=budget,
        session_id=SESSION_ID,
    )

    assert len(ctx2["conflicts"]) >= 1, (
        "Expected at least one conflict in context response after the conflicting entity write."
    )

    conflict_keys_in_ctx = [c["key"] for c in ctx2["conflicts"]]
    assert conflict_key in conflict_keys_in_ctx, (
        f"Expected conflict key '{conflict_key}' to appear in context conflicts, "
        f"but found: {conflict_keys_in_ctx}"
    )

    print(f"[Phase F] Conflict confirmed in context: key='{conflict_key}' "
          f"present in {conflict_keys_in_ctx}")

    # Final summary
    print(
        f"\n{'='*60}\n"
        f"E2E TEST PASSED\n"
        f"  Session:   {SESSION_ID}\n"
        f"  Events:    50 written, all 201\n"
        f"  Snapshots: {len(snapshots)}\n"
        f"  Entities:  {len(entity_map)}\n"
        f"  Context:   {len(ctx2['events'])} events, "
        f"{len(ctx2['snapshots'])} snapshots, "
        f"{len(ctx2['entities'])} entities, "
        f"{len(ctx2['conflicts'])} conflicts\n"
        f"{'='*60}"
    )
