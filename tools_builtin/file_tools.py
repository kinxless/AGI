"""File-system tools, restricted to WORKSPACE_ROOT.

All paths are resolved against WORKSPACE_ROOT and any attempt to escape it
(e.g. via '..' or absolute paths) raises ValueError. This keeps the agent
from reading or writing arbitrary files on the host.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent.tools import register_tool
from config import WORKSPACE_ROOT


def _resolve_inside_workspace(rel_path: str) -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    root = WORKSPACE_ROOT.resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"Path '{rel_path}' escapes the workspace root."
        ) from e
    return candidate


class ReadFileArgs(BaseModel):
    path: str = Field(..., description="Path relative to the workspace root.")


@register_tool(
    "read_file",
    "Read a UTF-8 text file from the workspace and return its contents.",
    ReadFileArgs,
)
def read_file(path: str) -> str:
    p = _resolve_inside_workspace(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    return p.read_text(encoding="utf-8")


class WriteFileArgs(BaseModel):
    path: str = Field(..., description="Path relative to the workspace root.")
    content: str = Field(..., description="UTF-8 text to write. Overwrites if exists.")


@register_tool(
    "write_file",
    "Write UTF-8 text to a file in the workspace, creating parent dirs as needed.",
    WriteFileArgs,
)
def write_file(path: str, content: str) -> str:
    p = _resolve_inside_workspace(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


class ListDirectoryArgs(BaseModel):
    path: str = Field(
        default=".",
        description="Directory path relative to the workspace root. Defaults to root.",
    )


@register_tool(
    "list_directory",
    "List files and subdirectories of a directory in the workspace.",
    ListDirectoryArgs,
)
def list_directory(path: str = ".") -> str:
    p = _resolve_inside_workspace(path)
    if not p.exists():
        raise FileNotFoundError(f"No such directory: {path}")
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")
    entries = []
    for child in sorted(p.iterdir()):
        kind = "dir" if child.is_dir() else "file"
        entries.append(f"{kind}: {child.name}")
    return "\n".join(entries) if entries else "(empty)"
