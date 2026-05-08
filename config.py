"""Central configuration for the agent.

All environment-specific values can be overridden via a .env file or shell
environment variables. This makes the same codebase run on Windows (local
Ollama) and Linux RunPod (GPU Ollama) without any file edits.

Priority: environment variable > .env file > default value below.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if present (no error if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR      = PROJECT_ROOT / "logs"
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")

OLLAMA = {
    "base_url":         os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "model":            os.environ.get("OLLAMA_MODEL",    "qwen2.5:7b-instruct"),
    "temperature":      float(os.environ.get("OLLAMA_TEMPERATURE", "0.2")),
    "request_timeout_s": int(os.environ.get("OLLAMA_TIMEOUT_S",   "300")),
}

KIMI = {
    "base_url":         "https://api.moonshot.ai/v1",
    "model":            os.environ.get("KIMI_MODEL", "kimi-k2"),
    "api_key_env":      "KIMI_API_KEY",
    "temperature":      0.2,
    "request_timeout_s": 120,
}

AGENT = {
    "max_iterations":      int(os.environ.get("AGENT_MAX_ITERATIONS", "20")),
    "max_json_retries":    3,
    "console_truncate_chars": 1200,
}

PYTHON_TOOL = {
    "timeout_s":      int(os.environ.get("PYTHON_TOOL_TIMEOUT_S", "15")),
    "max_output_chars": 8000,
}

# Path to the RAG repo root (the folder that contains app/rag.py).
# Set RAG_REPO_PATH in your .env or shell environment.
# Windows default kept so local dev still works without a .env file.
RAG_REPO_PATH = os.environ.get(
    "RAG_REPO_PATH",
    r"C:\Users\Civil\main projects",
)

PLANNER = {
    "step_max_attempts":   int(os.environ.get("STEP_MAX_ATTEMPTS",   "3")),
    "max_replan_attempts": int(os.environ.get("MAX_REPLAN_ATTEMPTS", "2")),
    "context_word_limit":  500,
    "max_approval_cycles": 3,
}
