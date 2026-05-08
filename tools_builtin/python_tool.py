"""Run short Python snippets in a subprocess with a hard timeout.

This is NOT a real sandbox. The child process inherits filesystem and network
access from the user. It only provides:
  - process isolation (a crash won't kill the agent)
  - a wall-clock timeout
  - stdout/stderr capture with size limits
  - working directory pinned to WORKSPACE_ROOT

Real isolation arrives in Phase 4 with Docker.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools import register_tool
from config import PYTHON_TOOL, WORKSPACE_ROOT


class ExecutePythonArgs(BaseModel):
    code: str = Field(..., description="Python source to execute. Use print() for output.")


@register_tool(
    "execute_python",
    (
        "Execute a Python snippet in a subprocess with a timeout. CWD is the "
        "agent workspace. Returns combined stdout+stderr and the exit code."
    ),
    ExecutePythonArgs,
)
def execute_python(code: str) -> str:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    timeout = PYTHON_TOOL["timeout_s"]
    max_chars = PYTHON_TOOL["max_output_chars"]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKSPACE_ROOT),
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = ""
        if out:
            combined += f"--- stdout ---\n{out}"
        if err:
            combined += f"\n--- stderr ---\n{err}"
        if not combined:
            combined = "(no output)"
        if len(combined) > max_chars:
            combined = combined[:max_chars] + f"\n... [truncated {len(combined) - max_chars} chars]"
        return f"exit_code={proc.returncode}\n{combined}"
    except subprocess.TimeoutExpired:
        return f"ERROR: execution exceeded timeout of {timeout}s and was killed."
    finally:
        try:
            script_path.unlink()
        except Exception:
            pass
