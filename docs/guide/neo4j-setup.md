# Neo4j Setup Guide

This project uses a dual-storage architecture:

- Supabase PostgreSQL for business data
- Neo4j for RAG vectors and graph data

## 1. Environment Variables

Set these values in `.env`:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=deeptutor_neo4j
NEO4J_DATABASE=neo4j
```

For local host access (outside Docker), you can use:

```env
NEO4J_URI=bolt://localhost:7687
```

## 2. Start Neo4j with Docker Compose

```bash
docker compose up -d neo4j
```

Check status:

```bash
docker compose ps neo4j
```

Expected:
- status is `healthy`
- Bolt `7687` and HTTP `7474` are exposed

## 3. Start Backend

The backend enforces a startup hard-check:

1. Neo4j connectivity
2. Neo4j schema/index bootstrap
3. Vector index availability

If Neo4j is not ready, backend startup fails immediately.

## 4. Data Location

Neo4j persistent volumes:

- `data/neo4j/data`
- `data/neo4j/logs`
- `data/neo4j/plugins`

## 5. Troubleshooting

- `connection refused`:
  - verify container is running and healthy
  - verify `NEO4J_URI` matches runtime network context
- `authentication failure`:
  - verify `NEO4J_USER` and `NEO4J_PASSWORD`
  - if changed, recreate container or update credentials in Neo4j
