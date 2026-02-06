# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepTutor 是一个 AI 驱动的个性化学习助手，采用多代理架构。它结合了 RAG（检索增强生成）、网络搜索和多代理协作，提供问题解答、问题生成、引导式学习和深度研究等功能。

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, LangChain + LangGraph, LlamaIndex
- **Frontend**: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS
- **Database**: PostgreSQL / SQLite（可配置）
- **LLM Providers**: OpenAI, Anthropic, Dashscope, Perplexity

## Common Commands

### Backend (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Install with all LLM providers
pip install -r requirements.txt -r .[all]

# Run tests
pytest tests/ --tb=short

# Lint with Ruff
ruff check src/

# Format with Ruff
ruff format src/

# Type checking
mypy src/

# Security linting
bandit -r src/

# Run backend
python scripts/start.py

# Run with frontend
python scripts/start_web.py

# Run FastAPI directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend (TypeScript)

```bash
cd web

# Install dependencies
npm install

# Development (with Turbopack)
npm run dev

# Build for production
npm run build
npm start

# Lint
npm run lint
```

### Docker

```bash
# Build and start
docker compose up

# Start with PostgreSQL
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
```

## Architecture

### Directory Structure

```
src/
├── agents/           # Agent modules (research, question, guide, etc.)
│   ├── base_agent.py # Unified BaseAgent class (所有代理的基类)
│   └── {module}/     # Individual agent modules
├── api/              # FastAPI backend
│   ├── routers/      # API endpoints
│   ├── main.py       # FastAPI app entry
│   └── utils/        # API utilities
├── services/         # Shared services
│   ├── llm/          # LLM orchestration (LangChain-based)
│   ├── rag/          # RAG services
│   ├── prompt/       # Prompt management
│   ├── storage/      # PostgreSQL/SQLite user data storage (NEW)
│   └── warmup.py     # Application warmup service
├── tools/            # Tool implementations
├── knowledge/        # Knowledge base management
└── utils/            # Helper utilities

web/
├── app/              # Next.js App Router pages
├── components/       # React components
├── context/          # Global state (GlobalContext.tsx)
└── lib/              # Utilities (api.ts)

config/
├── main.yaml         # Main system configuration
└── agents.yaml       # Unified agent parameters
```

### Key Architectural Patterns

1. **BaseAgent Pattern** (`src/agents/base_agent.py`)
   - All agents inherit from `BaseAgent`
   - LLM configuration via `AgentConfigResolver`
   - Prompt loading via `PromptManager`
   - LLM calls via `LLMOrchestrator` (streaming and non-streaming)

2. **Service Layer Architecture**
   - `AgentConfigResolver`: Configuration management
   - `LLMOrchestrator`: LLM call orchestration
   - `PromptManager`: YAML-based prompt loading
   - `StorageService`: PostgreSQL/SQLite user data persistence

3. **LangGraph Workflows** (Research Module)
   - Uses LangGraph for agent orchestration
   - Dynamic topic queues
   - Parallel/series execution modes

4. **Unified Configuration** (`config/agents.yaml`)
   - Single source of truth for agent parameters
   - Temperature, max_tokens per agent type

### API Endpoints

| Router | Key Endpoints |
|--------|--------------|
| `chat.py` | WebSocket chat |
| `guide.py` | Guided learning sessions |
| `knowledge.py` | KB management, document upload |
| `question.py` | Question generation |
| `research.py` | Research pipeline |
| `notebook.py` | Notebook CRUD |

- **Backend Port**: 8001
- **Frontend Port**: 3782
- **API Docs**: http://localhost:8001/docs

## Storage Backend (PostgreSQL)

The `feature/postgres-storage` branch adds PostgreSQL support:

```yaml
# config/main.yaml
storage:
  backend: postgres  # file | sqlite | postgres
  postgres_dsn: ""
  auto_migrate: true
```

Key files:
- `src/services/storage/postgres_db.py`: PostgreSQL backend
- `src/services/storage/user_db.py`: Unified user data interface

## Data Directory

```
data/
├── knowledge_bases/     # KB vector stores and documents
├── user/               # User-generated content
│   ├── solve/
│   ├── question/
│   ├── research/
│   ├── co-writer/
│   ├── guide/
│   ├── notebook/
│   └── logs/
└── db/                 # SQLite database
```

## Code Style (Python)

Configured in `pyproject.toml`:
- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff (E, F, I rules enabled)
- **Type Checker**: MyPy (relaxed mode)
- **Testing**: pytest with strict config

## Important Notes

1. **Configuration**: All agent parameters are in `config/agents.yaml`, not hardcoded
2. **Prompts**: Stored in YAML files per agent, loaded via `PromptManager`
3. **WebSocket Streaming**: Use `LogInterceptor` for streaming logs in WebSocket handlers
4. **LLM Calls**: All LLM calls should go through `LLMOrchestrator` for consistency
