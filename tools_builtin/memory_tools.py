"""Memory tools backed by the external RAG system.

Adds RAG_REPO_PATH to sys.path so the agent repo and RAG repo stay fully
independent — no code is copied, no symlinks needed.

If the import fails (wrong path, missing deps), a warning is printed and both
tools are skipped. The agent loop continues normally without them.
"""
from __future__ import annotations

import sys

from config import RAG_REPO_PATH

# --- attempt RAG import ---------------------------------------------------

_rag_available = False
_rag_error: str = ""

try:
    if RAG_REPO_PATH not in sys.path:
        sys.path.insert(0, RAG_REPO_PATH)
    from app.rag import add_text, vector_keyword_search  # type: ignore
    _rag_available = True
except Exception as e:
    _rag_error = str(e)
    print(
        f"\n[memory_tools] WARNING: RAG import failed — memory tools disabled.\n"
        f"  RAG_REPO_PATH = {RAG_REPO_PATH}\n"
        f"  Error: {e}\n"
        f"  Fix RAG_REPO_PATH in config.py and restart.\n"
    )

# --- only register tools if import succeeded ------------------------------

if _rag_available:
    from pydantic import BaseModel, Field
    from agent.tools import register_tool

    class SearchMemoryArgs(BaseModel):
        query: str = Field(..., description="Natural-language query to search memory.")
        max_results: int = Field(
            default=5, description="Maximum number of chunks to return (default 5)."
        )

    @register_tool(
        "search_memory",
        "Search persistent memory for information relevant to a query. Call this at the start of every task.",
        SearchMemoryArgs,
    )
    def search_memory(query: str, max_results: int = 5) -> str:
        try:
            results: list[str] = vector_keyword_search(query, k=max_results)
        except Exception as e:
            return f"search_memory error: {e}"
        if not results:
            return "No relevant memories found."
        separator = "\n---\n"
        return separator.join(str(r) for r in results)

    class SaveToMemoryArgs(BaseModel):
        text: str = Field(..., description="The text to store in memory.")
        source: str = Field(
            default="agent",
            description="Label for where this memory came from (e.g. 'agent', 'user').",
        )

    @register_tool(
        "save_to_memory",
        "Save a piece of text to persistent memory. Call this at the end of every task to store key results.",
        SaveToMemoryArgs,
    )
    def save_to_memory(text: str, source: str = "agent") -> str:
        try:
            add_text(text, source, collection_name="agent_memory")
            return "Saved to memory."
        except Exception as e:
            return f"save_to_memory error: {e}"
