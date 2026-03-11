# Neo4j Unified RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace local RAG storage with Neo4j-only retrieval/indexing for all three pipelines, keep Supabase for business data, and ship graph visualization in one release.

**Architecture:** Add a `graph_store` service layer that owns Neo4j schema, vector queries, and graph queries. Wire each pipeline to provider-specific adapters that emit a unified contract, then route all query paths through one orchestrator. Add startup hard-check and graph APIs consumed by a new frontend graph page.

**Tech Stack:** FastAPI, Python 3.10+, neo4j Python driver, existing RAG pipelines (llamaindex/lightrag/raganything), Next.js 16 + React 19 + Cytoscape.js, Docker Compose.

---

### Task 1: Add Neo4j Runtime and Configuration

**Files:**
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `config/main.yaml`
- Create: `docs/guide/neo4j-setup.md`

**Step 1: Write failing startup config test**

Create `tests/services/graph_store/test_config_required.py` asserting app config raises clear error if `NEO4J_URI/USER/PASSWORD` are absent.

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/graph_store/test_config_required.py -q`  
Expected: FAIL because graph store config module does not exist.

**Step 3: Add docker service and config keys**

Add `neo4j` service with healthcheck, ports `7474/7687`, persistent volumes, and auth env vars. Add config section and docs for required env.

**Step 4: Run test to verify it passes**

Run: `pytest tests/services/graph_store/test_config_required.py -q`  
Expected: PASS.

**Step 5: Commit**

Run:
```bash
git add docker-compose.yml README.md config/main.yaml docs/guide/neo4j-setup.md tests/services/graph_store/test_config_required.py
git commit -m "feat: add neo4j runtime config and docs"
```

### Task 2: Build Neo4j Base Client and Schema Bootstrap

**Files:**
- Create: `src/services/graph_store/neo4j_client.py`
- Create: `src/services/graph_store/schema.py`
- Create: `src/services/graph_store/types.py`
- Create: `tests/services/graph_store/test_schema_bootstrap.py`

**Step 1: Write failing schema bootstrap test**

Test should mock Neo4j session and assert constraints/index setup queries execute idempotently.

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/graph_store/test_schema_bootstrap.py -q`  
Expected: FAIL (module missing).

**Step 3: Implement minimal client and schema bootstrap**

Implement driver lifecycle (`connect/close/verify`) and schema initializer (`ensure_schema`) with fixed Cypher statements.

**Step 4: Run tests**

Run: `pytest tests/services/graph_store/test_schema_bootstrap.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/graph_store/neo4j_client.py src/services/graph_store/schema.py src/services/graph_store/types.py tests/services/graph_store/test_schema_bootstrap.py
git commit -m "feat: add neo4j client and schema bootstrap"
```

### Task 3: Implement Vector and Graph Repositories

**Files:**
- Create: `src/services/graph_store/vector_repo.py`
- Create: `src/services/graph_store/graph_repo.py`
- Create: `tests/services/graph_store/test_vector_repo.py`
- Create: `tests/services/graph_store/test_graph_repo.py`

**Step 1: Write failing repository contract tests**

Cover:
- chunk upsert with embedding
- vector top-k query
- subgraph query by anchor and hops

**Step 2: Run failing tests**

Run:
`pytest tests/services/graph_store/test_vector_repo.py tests/services/graph_store/test_graph_repo.py -q`  
Expected: FAIL.

**Step 3: Implement repositories**

Add explicit methods:
- `upsert_chunks(...)`
- `query_similar_chunks(...)`
- `upsert_entities_relations(...)`
- `fetch_subgraph(...)`

**Step 4: Run tests**

Run:
`pytest tests/services/graph_store/test_vector_repo.py tests/services/graph_store/test_graph_repo.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/graph_store/vector_repo.py src/services/graph_store/graph_repo.py tests/services/graph_store/test_vector_repo.py tests/services/graph_store/test_graph_repo.py
git commit -m "feat: add neo4j vector and graph repositories"
```

### Task 4: Add Provider Adapters for Three Pipelines

**Files:**
- Create: `src/services/graph_store/adapters/llamaindex_adapter.py`
- Create: `src/services/graph_store/adapters/lightrag_adapter.py`
- Create: `src/services/graph_store/adapters/raganything_adapter.py`
- Create: `tests/services/graph_store/test_provider_adapters.py`

**Step 1: Write failing adapter tests**

Cases:
- llamaindex writes chunk vector only
- lightrag writes graph relations
- raganything writes graph relations with evidence metadata

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/graph_store/test_provider_adapters.py -q`  
Expected: FAIL.

**Step 3: Implement adapters**

Add strict mapping into unified contract. Ensure llamaindex path does not call graph edge writes.

**Step 4: Run tests**

Run: `pytest tests/services/graph_store/test_provider_adapters.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/graph_store/adapters/*.py tests/services/graph_store/test_provider_adapters.py
git commit -m "feat: add provider-specific neo4j adapters"
```

### Task 5: Integrate Adapters into Pipeline Initialization

**Files:**
- Modify: `src/services/rag/pipelines/llamaindex.py`
- Modify: `src/services/rag/components/indexers/lightrag.py`
- Modify: `src/services/rag/pipelines/raganything.py`
- Create: `tests/services/rag/test_pipeline_neo4j_write.py`

**Step 1: Write failing integration tests**

Mock adapters and assert each pipeline calls its adapter after successful parse/index.

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/rag/test_pipeline_neo4j_write.py -q`  
Expected: FAIL.

**Step 3: Wire adapter calls**

Inject adapter invocation points at end of initialize/index process.

**Step 4: Run tests**

Run: `pytest tests/services/rag/test_pipeline_neo4j_write.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/rag/pipelines/llamaindex.py src/services/rag/components/indexers/lightrag.py src/services/rag/pipelines/raganything.py tests/services/rag/test_pipeline_neo4j_write.py
git commit -m "feat: write all pipeline artifacts into neo4j via adapters"
```

### Task 6: Add Unified Query Orchestrator and RAGService Wiring

**Files:**
- Create: `src/services/graph_store/query_orchestrator.py`
- Modify: `src/services/rag/service.py`
- Modify: `src/tools/rag_tool.py`
- Create: `tests/services/rag/test_neo4j_query_orchestrator.py`

**Step 1: Write failing orchestrator tests**

Verify unified return shape:
- `query/provider/mode`
- `retrieved_chunks[]`
- optional `graph_context`

**Step 2: Run test to verify it fails**

Run: `pytest tests/services/rag/test_neo4j_query_orchestrator.py -q`  
Expected: FAIL.

**Step 3: Implement orchestrator and service wiring**

Route all retrieval through Neo4j repositories, preserve external API shape in `rag_tool`.

**Step 4: Run tests**

Run: `pytest tests/services/rag/test_neo4j_query_orchestrator.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/graph_store/query_orchestrator.py src/services/rag/service.py src/tools/rag_tool.py tests/services/rag/test_neo4j_query_orchestrator.py
git commit -m "feat: route rag search through neo4j orchestrator"
```

### Task 7: Enforce Startup Hard-Check

**Files:**
- Modify: `src/api/main.py`
- Create: `tests/api/test_startup_requires_neo4j.py`

**Step 1: Write failing startup test**

Assert app startup aborts when Neo4j health check fails.

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_startup_requires_neo4j.py -q`  
Expected: FAIL.

**Step 3: Implement hard-check**

On startup:
1) connect
2) schema ensure
3) vector index verify  
Raise and terminate on failure.

**Step 4: Run tests**

Run: `pytest tests/api/test_startup_requires_neo4j.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/api/main.py tests/api/test_startup_requires_neo4j.py
git commit -m "feat: enforce neo4j startup hard-check"
```

### Task 8: Remove Local RAG Storage Readiness Dependencies

**Files:**
- Modify: `src/api/routers/knowledge.py`
- Modify: `src/knowledge/manager.py`
- Modify: `src/knowledge/initializer.py`
- Modify: `config/README.md`
- Create: `tests/knowledge/test_no_local_rag_storage_dependency.py`

**Step 1: Write failing tests**

Assert KB ready status no longer depends on local `rag_storage` directory presence.

**Step 2: Run test to verify it fails**

Run: `pytest tests/knowledge/test_no_local_rag_storage_dependency.py -q`  
Expected: FAIL.

**Step 3: Remove local dependency logic**

Switch ready checks to Neo4j index presence (or explicit status state).

**Step 4: Run tests**

Run: `pytest tests/knowledge/test_no_local_rag_storage_dependency.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/api/routers/knowledge.py src/knowledge/manager.py src/knowledge/initializer.py config/README.md tests/knowledge/test_no_local_rag_storage_dependency.py
git commit -m "refactor: remove local rag_storage readiness dependency"
```

### Task 9: Add Graph API Endpoints

**Files:**
- Create: `src/api/routers/graph.py`
- Modify: `src/api/main.py`
- Create: `tests/api/test_graph_router.py`

**Step 1: Write failing API tests**

Cover:
- `POST /api/v1/graph/query`
- `GET /api/v1/graph/subgraph`
- `POST /api/v1/graph/reindex`

**Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_graph_router.py -q`  
Expected: FAIL.

**Step 3: Implement graph router**

Validate inputs (`kb_name`, `hops`, `limit`) and call orchestrator/repositories.

**Step 4: Run tests**

Run: `pytest tests/api/test_graph_router.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/api/routers/graph.py src/api/main.py tests/api/test_graph_router.py
git commit -m "feat(api): add graph query and subgraph endpoints"
```

### Task 10: Frontend Graph Visualization Page

**Files:**
- Create: `web/app/graph/page.tsx`
- Create: `web/components/graph/GraphCanvas.tsx`
- Create: `web/components/graph/GraphFilters.tsx`
- Create: `web/components/graph/GraphDetails.tsx`
- Modify: `web/components/Sidebar.tsx`
- Create: `web/lib/graph.ts`
- Modify: `web/package.json`
- Create: `web/app/graph/__tests__/graph-page.test.tsx`

**Step 1: Write failing frontend tests**

Test:
- page loads
- search calls API
- renders nodes/edges
- shows truncation warning

**Step 2: Run tests to verify it fails**

Run: `cd web && npm run test -- graph-page.test.tsx`  
Expected: FAIL.

**Step 3: Implement page and components**

Use Cytoscape.js with debounce query and relation filter UI.

**Step 4: Run tests and lint**

Run:
- `cd web && npm run test -- graph-page.test.tsx`
- `cd web && npm run lint`  
Expected: tests PASS; lint may include pre-existing non-blocking warnings, but no new errors.

**Step 5: Commit**

```bash
git add web/app/graph web/components/graph web/components/Sidebar.tsx web/lib/graph.ts web/package.json web/app/graph/__tests__/graph-page.test.tsx
git commit -m "feat(web): add graph visualization page"
```

### Task 11: End-to-End Verification and Release Readiness

**Files:**
- Create: `tests/integration/test_neo4j_rag_e2e.py`
- Modify: `docs/guide/troubleshooting.md`

**Step 1: Write e2e test**

Flow:
1) init KB via one provider
2) query RAG
3) query subgraph
4) verify unified contract fields

**Step 2: Run full verification**

Run:
- `pytest tests/integration/test_neo4j_rag_e2e.py -q`
- `pytest tests/api/test_route_protection.py -q`
- `pytest tests/api/test_graph_router.py -q`
- `pytest tests/services/rag/test_neo4j_query_orchestrator.py -q`
- `cd web && npm run lint`

Expected:
- backend tests PASS
- frontend lint has no new errors introduced by this change

**Step 3: Commit**

```bash
git add tests/integration/test_neo4j_rag_e2e.py docs/guide/troubleshooting.md
git commit -m "test: add neo4j rag e2e coverage and docs"
```

### Task 12: Final Cleanup for Single-Cutover

**Files:**
- Modify: `README.md`
- Modify: `docs/guide/data-preparation.md`

**Step 1: Remove local storage references in docs**

Update docs so RAG requires Neo4j and no longer documents local `rag_storage` as runtime dependency.

**Step 2: Verify git diff scope**

Run: `git diff --name-only main...HEAD`  
Expected: only intended backend/frontend/docs/test files.

**Step 3: Final commit**

```bash
git add README.md docs/guide/data-preparation.md
git commit -m "docs: finalize neo4j-only rag cutover guidance"
```

## Notes for Execution

- Do not modify unrelated pre-existing lint issues unless they block the build for touched files.
- Keep provider adapter behavior explicit:
  - `llamaindex` must not create `Entity/REL` edges.
- Keep response contract stable for existing consumers (`rag_tool`, chat sources).
- Prefer small commits per task for easier rollback and review.

