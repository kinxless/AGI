"""Planner: turns a task string into an ordered list of Steps.

Makes its own LLM call (separate from the execution loop) so the planning
prompt can be specialised without polluting the agent system prompt.
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from agent.llm import LLMClient, get_llm_client
from agent.logger import NullLogger, RunLogger


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Step(BaseModel):
    step_number: int
    description: str
    tools_likely_needed: List[str] = Field(default_factory=list)
    depends_on: List[int] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    result: Optional[str] = None
    attempts: int = 0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a planning assistant. Given a task, produce a concrete, ordered execution plan.

Output ONLY a JSON object with the key "steps", whose value is a list of step objects.
Each step object has exactly these fields:
  - step_number: integer starting at 1
  - description: string — what to do in plain English
  - tools_likely_needed: list of strings — tool names likely required
  - depends_on: list of integers — step numbers that must complete before this one

No prose, no markdown fences. Output ONLY the JSON object.

Available tools: {tool_list}
"""

_REPLAN_SYSTEM = """\
You are a planning assistant. A step in the execution plan has failed.
Your job is to produce a revised plan for the REMAINING work only.
Completed steps must NOT appear in the new plan.

Output ONLY a JSON object with the key "steps" (same schema as before).
No prose, no markdown fences.

Available tools: {tool_list}
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    def __init__(
        self,
        llm: LLMClient | None = None,
        logger: RunLogger | NullLogger | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.log = logger or NullLogger()

    def _tool_list(self) -> str:
        from agent.tools import all_tools
        return ", ".join(t.name for t in all_tools()) or "(none)"

    def _parse_steps(self, raw: str, number_offset: int = 0) -> List[Step]:
        """Extract a steps list from raw LLM output."""
        # Strip markdown fences if present
        text = re.sub(r"```[a-z]*", "", raw).strip()
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in planner output.")
        data: dict[str, Any] = json.loads(text[start:])
        raw_steps = data.get("steps", [])
        steps: List[Step] = []
        for i, s in enumerate(raw_steps, start=1):
            steps.append(Step(
                step_number=number_offset + i,
                description=s.get("description", f"Step {i}"),
                tools_likely_needed=s.get("tools_likely_needed", []),
                depends_on=[number_offset + d for d in s.get("depends_on", [])],
            ))
        return steps

    def plan(self, task: str, memory_context: str = "") -> List[Step]:
        context_block = (
            f"\nRelevant memories:\n{memory_context}\n" if memory_context.strip() else ""
        )
        user_msg = f"{context_block}Task: {task}"
        system = _PLAN_SYSTEM.format(tool_list=self._tool_list())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        self.log.info(f"PLANNER: generating plan for task: {task}")
        raw = self.llm.chat(messages)
        self.log.info(f"PLANNER raw output:\n{raw}")
        steps = self._parse_steps(raw)
        self.log.info(f"PLANNER: produced {len(steps)} steps")
        return steps

    def replan(
        self,
        task: str,
        original_steps: List[Step],
        failed_step: Step,
        error: str,
    ) -> List[Step]:
        done_lines = []
        for s in original_steps:
            if s.status == "done":
                result_snippet = (s.result or "")[:120]
                done_lines.append(f"  Step {s.step_number}: {s.description} → {result_snippet}")
        done_summary = "\n".join(done_lines) or "  (none)"

        number_offset = max((s.step_number for s in original_steps), default=0)

        user_msg = (
            f"Task: {task}\n\n"
            f"Completed steps:\n{done_summary}\n\n"
            f"Failed step {failed_step.step_number}: {failed_step.description}\n"
            f"Error: {error}\n\n"
            "Produce a revised plan for the remaining work only. "
            "Do not include already-completed steps."
        )
        system = _REPLAN_SYSTEM.format(tool_list=self._tool_list())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        self.log.info(f"PLANNER: replanning after failure of step {failed_step.step_number}")
        raw = self.llm.chat(messages)
        self.log.info(f"PLANNER replan raw output:\n{raw}")
        new_steps = self._parse_steps(raw, number_offset=number_offset)
        self.log.info(f"PLANNER: replan produced {len(new_steps)} new steps")
        return new_steps

    def display_plan(self, steps: List[Step]) -> str:
        lines = ["", "=" * 60, "  PROPOSED PLAN", "=" * 60]
        for s in steps:
            tools = ", ".join(s.tools_likely_needed) if s.tools_likely_needed else "—"
            deps = ", ".join(str(d) for d in s.depends_on) if s.depends_on else "none"
            lines.append(f"\n  Step {s.step_number}: {s.description}")
            lines.append(f"    Tools : {tools}")
            lines.append(f"    Needs : steps {deps}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
