"""The core agent loop: observe -> think -> act -> repeat.

The model returns a single JSON object per turn with one of two shapes:

    {"thought": "...", "action": {"tool": "<name>", "args": {...}}}
    {"thought": "...", "final_answer": "..."}

The loop validates, executes the tool, and feeds the result back as the next
user message. Malformed JSON is retried up to AGENT["max_json_retries"] times
before the iteration counts against the budget.
"""
from __future__ import annotations

import json
from typing import Any

from config import AGENT
from agent.llm import LLMClient, extract_json_block, get_llm_client
from agent.logger import RunLogger
from agent.tools import (
    ToolArgError,
    ToolNotFoundError,
    get_tool,
    render_tool_catalog,
)


SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent that solves the user's task by calling tools.

On every turn you MUST output a single JSON object and nothing else (no prose,
no markdown fences). The JSON has one of two shapes:

  Shape A (call a tool):
    {{"thought": "<short reasoning>", "action": {{"tool": "<tool_name>", "args": {{...}}}}}}

  Shape B (finish):
    {{"thought": "<short reasoning>", "final_answer": "<answer to the user>"}}

Example of Shape A (correct — one object, thought and action together):
    {{"thought": "I should search memory first.", "action": {{"tool": "search_memory", "args": {{"query": "user preferences"}}}}}}

WRONG (two separate objects — never do this):
    {{"thought": "..."}} {{"action": {{...}}}}

Rules:
- Output ONLY the JSON object. No surrounding text.
- "thought" is a brief private note about your next step.
- Use Shape B as soon as you have enough information to answer the user.
- If a tool returns an error observation, read it carefully and try a different
  approach instead of repeating the same call.

Memory:
- You have persistent memory across sessions via search_memory and save_to_memory.
- At the START of every task, call search_memory with the task as the query before doing anything else.
- At the END of every task, call save_to_memory to store the key result, decision, or learning.
- If search_memory or save_to_memory are unavailable, skip silently and continue.

Available tools:
{tool_catalog}
"""


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient | None = None,
        logger: RunLogger | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.log = logger or RunLogger()
        self.max_iters: int = AGENT["max_iterations"]
        self.max_retries: int = AGENT["max_json_retries"]

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(tool_catalog=render_tool_catalog())

    def run(self, task: str) -> str:
        system_prompt = self._build_system_prompt()
        self.log.system(system_prompt)
        self.log.user_task(task)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]

        for iteration in range(1, self.max_iters + 1):
            parsed = self._call_with_retry(iteration, messages)
            if parsed is None:
                msg = "Model failed to produce valid JSON after retries; aborting."
                self.log.info(msg)
                return msg

            messages.append({"role": "assistant", "content": _compact_json(parsed)})

            if "final_answer" in parsed:
                final = str(parsed["final_answer"])
                self.log.done(iteration, final)
                return final

            action = parsed.get("action") or {}
            tool_name = action.get("tool", "")
            tool_args = action.get("args", {}) or {}
            observation = self._execute_tool(iteration, tool_name, tool_args)
            messages.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        msg = f"Max iterations ({self.max_iters}) reached without a final answer."
        self.log.info(msg)
        return msg

    def _call_with_retry(
        self, iteration: int, messages: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        attempt_msgs = list(messages)
        for attempt in range(1, self.max_retries + 1):
            self.log.model_input(iteration, attempt_msgs)
            raw = self.llm.chat(attempt_msgs)
            self.log.model_output(iteration, raw)
            try:
                parsed = extract_json_block(raw)
            except Exception as e:
                self.log.parse_error(iteration, attempt, str(e), raw)
                attempt_msgs = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON matching the "
                            "required schema. Output ONLY a single JSON object "
                            "with the shape described in the system prompt. "
                            f"Parser error: {e}"
                        ),
                    },
                ]
                continue

            if not _has_valid_shape(parsed):
                err = (
                    "JSON did not match required shape. Must contain either "
                    "'final_answer' or an 'action' with 'tool' and 'args'."
                )
                self.log.parse_error(iteration, attempt, err, raw)
                attempt_msgs = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": err},
                ]
                continue

            return parsed
        return None

    def _execute_tool(self, iteration: int, name: str, args: dict[str, Any]) -> str:
        self.log.tool_call(iteration, name, args)
        try:
            tool = get_tool(name)
            result = tool.call(args)
            self.log.tool_result(iteration, name, ok=True, result=result)
            return f"[{name}] {result}"
        except ToolNotFoundError:
            err = (
                f"Tool '{name}' is not registered. Use one of the tools listed "
                "in the system prompt."
            )
            self.log.tool_result(iteration, name, ok=False, result=err)
            return f"ERROR: {err}"
        except ToolArgError as e:
            err = f"Invalid arguments for tool '{name}': {e}"
            self.log.tool_result(iteration, name, ok=False, result=err)
            return f"ERROR: {err}"
        except Exception as e:
            err = f"Tool '{name}' raised {type(e).__name__}: {e}"
            self.log.tool_result(iteration, name, ok=False, result=err)
            return f"ERROR: {err}"


def _has_valid_shape(parsed: dict[str, Any]) -> bool:
    if "final_answer" in parsed:
        return True
    action = parsed.get("action")
    if not isinstance(action, dict):
        return False
    return isinstance(action.get("tool"), str) and isinstance(
        action.get("args", {}), dict
    )


def _compact_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)
