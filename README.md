# AGI Project

A phased agentic AI assistant built on a local Ollama model, with a
decorator-based tool registry, persistent RAG memory, and a hierarchical
planner layer. Designed so each phase slots in without touching the layers
below it.

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Agent loop, tool registry, Ollama backend |
| 2 | ✅ Done | Persistent memory via RAG (ChromaDB + SentenceTransformers) |
| 3 | ✅ Done | Hierarchical planner with approval flow and auto-replanning |
| 4 | Planned | Computer use in a Docker sandbox |
| 5 | Planned | Swap backend to Kimi K2 via API |

---

## Architecture

```
main.py                    # CLI entry point
config.py                  # All config — env vars override everything

agent/
  orchestrator.py          # Phase 3 top-level: plan → approve → execute → memory
  planner.py               # Breaks tasks into Steps, replans on failure
  executor.py              # Runs one Step through the agent loop (max 3 retries)
  loop.py                  # Core observe → think → act loop
  tools.py                 # @register_tool decorator + Pydantic-validated args
  llm.py                   # LLMClient ABC, OllamaClient, KimiClient (stub)
  logger.py                # Console (truncated) + JSONL transcript (full)

tools_builtin/
  file_tools.py            # read_file, write_file, list_directory
  python_tool.py           # execute_python (subprocess + timeout)
  memory_tools.py          # search_memory, save_to_memory (via RAG repo)

workspace/                 # Agent's sandboxed file workspace (auto-created)
logs/                      # Per-run JSONL transcripts (auto-created)
```

### How a task flows (Phase 3)

```
User task
  └─ Orchestrator.run()
       ├─ search_memory(task)          # inject relevant past context
       ├─ Planner.plan(task)           # LLM call → List[Step]
       ├─ User approval loop           # show plan, accept / reject / edit
       ├─ StepExecutor.execute_step()  # calls AgentLoop per step, up to 3 retries
       │    └─ on failure → Planner.replan() → user approval → continue
       ├─ save_to_memory(summary)      # persist results
       └─ final report
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- The [RAG repo](https://github.com/kinxless/RAG_system) cloned locally

### 1. Clone and configure

```bash
git clone <this-repo>
cd AGI-project
cp .env.example .env
# Edit .env — set RAG_REPO_PATH at minimum
```

### 2. Install Ollama and pull the model

**Windows:**
```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:7b-instruct
ollama serve
```

**Linux / RunPod:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:7b-instruct
ollama serve &
```

### 3. Python environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install RAG repo dependencies too:
```bash
pip install -r /path/to/rag_repo/requirements.txt
```

### 4. RunPod (one command)

```bash
export RAG_REPO_URL=https://github.com/kinxless/RAG_system
bash setup_runpod.sh
source .venv/bin/activate
```

---

## Configuration

All values can be set in `.env` (copy from `.env.example`). No code edits needed
between environments.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` or `kimi` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model to use |
| `OLLAMA_TIMEOUT_S` | `300` | Request timeout in seconds |
| `RAG_REPO_PATH` | *(Windows default)* | Absolute path to RAG repo root |
| `KIMI_API_KEY` | — | Required if `LLM_BACKEND=kimi` |
| `AGENT_MAX_ITERATIONS` | `20` | Max loop iterations per step |
| `STEP_MAX_ATTEMPTS` | `3` | Retries per planner step |
| `MAX_REPLAN_ATTEMPTS` | `2` | How many times replanning is allowed |

---

## Usage

### Standard (Phase 3 — planner + memory)

```bash
python main.py "your task here"
```

The agent will:
1. Search memory for relevant past context
2. Generate a step-by-step plan and show it to you
3. Wait for your approval (`yes` / `no` / `edit`)
4. Execute each step, replanning automatically on failure
5. Save a summary to memory
6. Print a final report

### Direct loop (bypass planner)

```bash
python main.py --no-plan "your task here"
```

Useful for quick single-step tasks or debugging.

---

## Adding a tool

```python
# tools_builtin/my_tool.py
from pydantic import BaseModel, Field
from agent.tools import register_tool

class MyArgs(BaseModel):
    input: str = Field(..., description="What to process.")

@register_tool("my_tool", "One-line description for the model.", MyArgs)
def my_tool(input: str) -> str:
    return f"processed: {input}"
```

Then add it to `tools_builtin/__init__.py`:
```python
from . import my_tool  # noqa: F401
```

The tool is automatically added to the system prompt on the next run.

---

## Logs

Every run writes `logs/run_<timestamp>.jsonl` — full system prompt, every model
input/output, every tool call and result, every parse error, untruncated.
The console shows the same stream truncated to 1200 chars per block.

---

## Switching to Kimi (Phase 5)

```env
LLM_BACKEND=kimi
KIMI_API_KEY=your_key_here
```

`KimiClient` uses an OpenAI-compatible chat completions endpoint. No other
changes needed — the agent loop, tools, planner, and orchestrator are all
backend-agnostic.
