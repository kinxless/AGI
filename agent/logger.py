"""Verbose logger for the agent.

Writes a pretty, truncated stream to the console and a full JSONL transcript
to logs/run_<timestamp>.jsonl. Every model input, model output, tool call,
and tool result passes through here.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import AGENT, LOG_DIR

_SEP = "=" * 78
_SUB = "-" * 78


class RunLogger:
    def __init__(self, run_name: str | None = None) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_name}" if run_name else ""
        self.path = LOG_DIR / f"run_{ts}{suffix}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.time()
        self._truncate = AGENT["console_truncate_chars"]

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def _write_jsonl(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "t": round(time.time() - self._t0, 3),
            "kind": kind,
            **payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"

    def _console(self, header: str, body: str) -> None:
        print(f"\n{_SEP}\n{header}\n{_SUB}", file=sys.stderr)
        print(self._truncate_text(body, self._truncate), file=sys.stderr)
        print(_SEP, file=sys.stderr, flush=True)

    def system(self, prompt: str) -> None:
        self._console("SYSTEM PROMPT", prompt)
        self._write_jsonl("system_prompt", {"prompt": prompt})

    def user_task(self, task: str) -> None:
        self._console("USER TASK", task)
        self._write_jsonl("user_task", {"task": task})

    def model_input(self, iteration: int, messages: list[dict[str, Any]]) -> None:
        body = json.dumps(messages, ensure_ascii=False, indent=2, default=str)
        self._console(f"MODEL INPUT  (iter {iteration})", body)
        self._write_jsonl("model_input", {"iteration": iteration, "messages": messages})

    def model_output(self, iteration: int, raw: str) -> None:
        self._console(f"MODEL OUTPUT (iter {iteration})", raw)
        self._write_jsonl("model_output", {"iteration": iteration, "raw": raw})

    def parse_error(self, iteration: int, attempt: int, error: str, raw: str) -> None:
        self._console(
            f"PARSE ERROR  (iter {iteration}, retry {attempt})",
            f"{error}\n--- raw ---\n{raw}",
        )
        self._write_jsonl(
            "parse_error",
            {"iteration": iteration, "attempt": attempt, "error": error, "raw": raw},
        )

    def tool_call(self, iteration: int, name: str, args: dict[str, Any]) -> None:
        body = f"{name}({json.dumps(args, ensure_ascii=False, default=str)})"
        self._console(f"TOOL CALL    (iter {iteration})", body)
        self._write_jsonl("tool_call", {"iteration": iteration, "name": name, "args": args})

    def tool_result(self, iteration: int, name: str, ok: bool, result: Any) -> None:
        body = f"ok={ok}\n{result}"
        self._console(f"TOOL RESULT  (iter {iteration}) [{name}]", body)
        self._write_jsonl(
            "tool_result",
            {"iteration": iteration, "name": name, "ok": ok, "result": result},
        )

    def done(self, iteration: int, final: str) -> None:
        self._console(f"DONE         (iter {iteration})", final)
        self._write_jsonl("done", {"iteration": iteration, "final": final})

    def info(self, msg: str) -> None:
        self._console("INFO", msg)
        self._write_jsonl("info", {"msg": msg})


class NullLogger:
    """No-op logger used as the default in internal classes.

    Lets Planner, StepExecutor, and Orchestrator call self.log.info()
    unconditionally without None guards — output is silently discarded
    unless a real RunLogger is injected.
    """
    path: Path | None = None

    def close(self) -> None: pass
    def system(self, prompt: str) -> None: pass
    def user_task(self, task: str) -> None: pass
    def model_input(self, iteration: int, messages: list) -> None: pass
    def model_output(self, iteration: int, raw: str) -> None: pass
    def parse_error(self, iteration: int, attempt: int, error: str, raw: str) -> None: pass
    def tool_call(self, iteration: int, name: str, args: dict) -> None: pass
    def tool_result(self, iteration: int, name: str, ok: bool, result: object) -> None: pass
    def done(self, iteration: int, final: str) -> None: pass
    def info(self, msg: str) -> None: pass
