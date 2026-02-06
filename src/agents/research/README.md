# DR-in-KG 2.0: Deep Research System

> A systematic deep research system based on **Dynamic Topic Queue** architecture, enabling multi-agent collaboration across three phases: **Planning -> Researching -> Reporting**.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Three-Phase Pipeline](#three-phase-pipeline)
- [Core Data Structures](#core-data-structures)
- [Agent Responsibilities](#agent-responsibilities)
- [Tool Integration](#tool-integration)
- [Citation System](#citation-system)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Architecture Overview

```
User Input Topic
    |
    v
+-----------------------------------------------------------+
|  Phase 1: Planning                                        |
|  - RephraseAgent: Topic optimization (with user feedback)|
|  - DecomposeAgent: Subtopic decomposition (RAG-enhanced) |
|  - Initialize DynamicTopicQueue                          |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|  Phase 2: Researching (Dynamic Loop)                     |
|  - ManagerAgent: Queue scheduling & task distribution       |
|  - ResearchAgent: Sufficiency check & query planning        |
|  - Tool Execution: RAG / Web / Paper / Code              |
|  - NoteAgent: Information compression & ToolTrace creation |
|                                                             |
|  Execution Modes: Series (sequential) | Parallel          |
+-----------------------------------------------------------+
    |
    v
+-----------------------------------------------------------+
|  Phase 3: Reporting                                       |
|  - Deduplication: Remove redundant topics                 |
|  - Outline Generation: Three-level heading structure      |
|  - Report Writing: Markdown with inline citations         |
+-----------------------------------------------------------+
    |
    v
Final Research Report (Markdown)
```

### Directory Structure

```
src/agents/research/
├── data_structures.py       # TopicBlock, ToolTrace, DynamicTopicQueue
├── graph/                   # LangGraph-based orchestrator
│   ├── graph.py            # StateGraph definition
│   ├── nodes.py            # Node functions for each phase
│   └── state.py            # State schema
├── agents/
│   ├── rephrase_agent.py    # Topic optimization
│   ├── decompose_agent.py   # Subtopic decomposition
│   ├── manager_agent.py     # Queue management
│   ├── research_agent.py    # Research decisions
│   ├── note_agent.py       # Information compression
│   └── reporting_agent.py    # Report generation
├── prompts/
│   ├── en/                 # English prompts
│   └── zh/                 # Chinese prompts
└── utils/
    ├── citation_manager.py   # Citation ID management
    ├── json_utils.py        # JSON parsing utilities
    └── token_tracker.py     # Token usage tracking
```

---

## Quick Start

### CLI Usage

```bash
# Using the CLI launcher with LangGraph orchestrator
python scripts/start.py

# Select option 2 for deep research
```

### WebSocket API

```bash
# Connect to WebSocket and send research request
ws://localhost:8000/api/v1/research/run

{
  "topic": "Deep Learning Basics",
  "kb_name": "ai_textbook",
  "plan_mode": "medium",
  "enabled_tools": ["RAG", "Paper", "Web"],
  "skip_rephrase": false
}
```

### Python API

```python
import asyncio
from src.agents.research.graph import build_research_graph
from src.services.config import load_config_with_main
from datetime import datetime

async def main():
    # Load configuration
    config = load_config_with_main("main.yaml")

    # Build LangGraph
    graph = build_research_graph()

    # Prepare initial state
    initial_state = {
        "topic": "Attention Mechanisms",
        "research_id": f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "kb_name": "ai_textbook",
        "config": config,
    }

    # Run research
    result = await graph.ainvoke(initial_state)
    print(f"Report: {result['result']['final_report_path']}")

asyncio.run(main())
```

---

## Three-Phase Pipeline

### Phase 1: Planning

**Goal**: Transform user input into a structured research plan with subtopics.

#### 1.1 RephraseAgent (Topic Optimization)

- **Input**: User's original topic
- **Process**:
  1. Analyze and optimize the topic for research
  2. Support multi-turn user interaction for refinement
  3. LLM judges user satisfaction to decide continuation
- **Output**: Optimized research topic with clear focus and scope

```json
{
  "topic": "Optimized, specific, and researchable topic description (200-400 words)"
}
```

#### 1.2 DecomposeAgent (Subtopic Decomposition)

- **Input**: Optimized topic, target number of subtopics
- **Process**:
  1. Generate 3-8 subtopics covering different aspects of the topic
  2. Provide a brief overview for each subtopic
  3. Optionally retrieve relevant context using RAG
- **Output**: List of subtopics with overviews

```json
{
  "sub_topics": [
    {
      "title": "Subtopic 1",
      "description": "Brief description of subtopic 1"
    }
  ],
  "total_subtopics": 5
}
```

#### 1.3 DynamicTopicQueue Initialization

- **Process**: Convert subtopics into TopicBlock objects with unique IDs
- **State**: Initialize queue with all blocks in PENDING status
- **Output**: Ready-to-research topic queue

---

### Phase 2: Researching

**Goal**: Execute iterative research on each subtopic using multiple tools.

#### 2.1 ResearchAgent

The core agent that coordinates the research loop:

1. **Sufficiency Check**: Determines if current knowledge is sufficient
   - For "fixed" iteration mode: Use conservative criteria
   - For "flexible" mode: Agent decides when to stop

2. **Query Planning**: Generate next research query
   - Analyze current knowledge gaps
   - Select appropriate tool (RAG, Web, Paper, Code)
   - Generate search query

3. **Tool Execution**: Execute the planned tool call
   - RAG: Knowledge base search
   - Web Search: Real-time web information
   - Paper Search: Academic papers from arXiv
   - Code Execution: Run Python calculations

#### 2.2 NoteAgent

- **Input**: Raw tool output
- **Process**:
  1. Extract key information
  2. Generate summary
  3. Create ToolTrace
- **Output**: Structured ToolTrace with summary and metadata

#### 2.3 ManagerAgent

- **Queue Management**: Track pending/completed/failed blocks
- **Topic Discovery**: Optionally add new topics discovered during research
- **Statistics**: Track tool calls, iterations, progress

---

### Phase 3: Reporting

**Goal**: Generate comprehensive research report.

#### 3.1 ReportingAgent

1. **Outline Generation**: Create three-level heading structure
2. **Content Writing**: Generate detailed content for each section
3. **Citation Integration**: Insert inline citations using collected ToolTraces
4. **Final Polish**: Formatting, references, table of contents

---

## Core Data Structures

### TopicBlock

```python
class TopicBlock:
    block_id: str           # Unique identifier (e.g., "topic_0")
    sub_topic: str          # Research subtopic
    overview: str           # Brief overview
    status: TopicStatus     # PENDING, RESEARCHING, COMPLETED, FAILED
    tool_traces: list[ToolTrace]  # Collected tool outputs
    iteration_count: int    # Number of research iterations
```

### ToolTrace

```python
class ToolTrace:
    tool_type: str          # rag_hybrid, web_search, etc.
    query: str              # Search query
    summary: str            # Generated summary
    raw_answer: str         # Original tool output
    citation_id: str        # Unique citation ID
    timestamp: datetime     # When tool was executed
```

### DynamicTopicQueue

```python
class DynamicTopicQueue:
    blocks: list[TopicBlock]    # All topic blocks
    completed: int              # Count of completed blocks
    pending: int                # Count of pending blocks
    failed: int                 # Count of failed blocks

    def add_block(sub_topic, overview) -> TopicBlock
    def get_next_task() -> Optional[TopicBlock]
    def complete_task(block_id)
    def fail_task(block_id, error)
    def get_statistics() -> dict
```

---

## Agent Responsibilities

| Agent | Input | Output | Key Methods |
|-------|-------|--------|-------------|
| RephraseAgent | Raw topic | Optimized topic | process() |
| DecomposeAgent | Optimized topic, num_subtopics | Subtopic list | process() |
| ResearchAgent | TopicBlock, current_knowledge | Updated TopicBlock | process(), check_sufficiency(), generate_query_plan() |
| NoteAgent | Tool output | ToolTrace | process() |
| ManagerAgent | Queue state | Updated queue | complete_task(), fail_task() |
| ReportingAgent | Queue, topic | Final report | process() |

---

## Tool Integration

### Supported Tools

| Tool | Purpose | Configuration Key |
|------|---------|-------------------|
| rag_hybrid | Knowledge base search (hybrid) | enable_rag_hybrid |
| rag_naive | Knowledge base search (basic) | enable_rag_naive |
| query_item | Query specific numbered items | enable_query_item |
| web_search | Real-time web search | enable_web_search |
| paper_search | Academic paper search | enable_paper_search |
| run_code | Python code execution | enable_run_code |

### Tool Selection Strategy

- **Phase 1 (Early)**: Use RAG tools for foundational knowledge
- **Phase 2 (Middle)**: Introduce Paper/Web search for depth
- **Phase 3 (Late)**: Use all tools to fill gaps, verify with code

---

## Citation System

- Each tool call gets a unique citation ID (e.g., "CIT-0-01")
- Citations are tracked in ToolTrace
- Report citations reference these IDs
- Citation manager handles ID generation and tracking

---

## Configuration

### Configuration Files

- `main.yaml`: Main configuration (paths, RAG settings)
- `agents.yaml`: Agent-specific parameters (temperature, max_tokens)

### Key Configuration Options

```yaml
planning:
  rephrase:
    enabled: true
    max_iterations: 3
  decompose:
    initial_subtopics: 5
    mode: manual  # or "auto"

researching:
  max_iterations: 5
  iteration_mode: fixed  # or "flexible"
  enable_rag_hybrid: true
  enable_web_search: false
  enable_paper_search: false

reporting:
  min_section_length: 500
  enable_inline_citations: true
```

### Plan Modes

| Mode | Subtopics | Iterations | Iteration Mode |
|------|-----------|------------|----------------|
| quick | 2 | 2 | fixed |
| medium | 5 | 4 | fixed |
| deep | 8 | 7 | fixed |
| auto | auto | 6 | flexible |

---

## Output Files

All outputs are saved to `data/user/research/`:

```
data/user/research/
├── cache/
│   └── <research_id>/
│       ├── planning_progress.json
│       ├── researching_progress.json
│       ├── queue.json
│       └── langgraph_checkpoints/  # If checkpoint enabled
└── reports/
    └── <research_id>.md            # Final report
    └── <research_id>_metadata.json # Metadata
```

### Report Structure

```markdown
# <Research Topic>

## Table of Contents

## 1. <Subtopic 1>
### 1.1 <Aspect>
Content with inline citations [CIT-0-01]

## References

[1] Tool trace reference
```

---

## License

Apache-2.0
