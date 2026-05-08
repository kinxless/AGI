"""StepExecutor: runs a single Step through the existing AgentLoop.

The loop is treated as a black box — executor.py calls loop.run() and reads
the return value. No modifications to loop.py are needed or made.
"""
from __future__ import annotations

import textwrap

from config import PLANNER
from agent.llm import LLMClient
from agent.logger import RunLogger
from agent.loop import AgentLoop
from agent.planner import Step

_MAX_ATTEMPTS: int = PLANNER["step_max_attempts"]


class StepExecutor:
    def __init__(
        self,
        llm: LLMClient | None = None,
        logger: RunLogger | None = None,
    ) -> None:
        self.llm = llm
        self.log = logger

    def _log_info(self, msg: str) -> None:
        if self.log:
            self.log.info(msg)

    def execute_step(self, step: Step, context: str) -> str:
        """Run step up to _MAX_ATTEMPTS times. Returns the result string.

        Raises StepFailed if all attempts are exhausted.
        """
        last_error = ""
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            step.attempts = attempt
            step.status = "running"

            task_prompt = _build_step_prompt(step, context, attempt, last_error)
            self._log_info(
                f"EXECUTOR: step {step.step_number} attempt {attempt}/{_MAX_ATTEMPTS}"
            )

            loop = AgentLoop(llm=self.llm, logger=self.log)
            try:
                result = loop.run(task_prompt)
            except Exception as e:
                result = f"AgentLoop raised {type(e).__name__}: {e}"

            failed_keywords = ("failed", "error", "could not", "unable", "exception")
            if _looks_like_failure(result, failed_keywords):
                last_error = result
                self._log_info(
                    f"EXECUTOR: step {step.step_number} attempt {attempt} looks failed: "
                    f"{result[:200]}"
                )
                if attempt == _MAX_ATTEMPTS:
                    step.status = "failed"
                    step.result = result
                    raise StepFailed(
                        f"Step {step.step_number} failed after {_MAX_ATTEMPTS} attempts. "
                        f"Last error: {result}"
                    )
                continue

            step.status = "done"
            step.result = result
            self._log_info(
                f"EXECUTOR: step {step.step_number} done on attempt {attempt}"
            )
            return result

        # Unreachable but satisfies type checker
        raise StepFailed(f"Step {step.step_number}: exhausted attempts")


class StepFailed(Exception):
    """Raised when a step exhausts all retry attempts."""


def _build_step_prompt(
    step: Step, context: str, attempt: int, last_error: str
) -> str:
    parts = []
    if context.strip():
        parts.append(f"Context from previous steps:\n{context}\n")
    parts.append(f"Your current task: {step.description}")
    if attempt > 1 and last_error:
        hint = textwrap.shorten(last_error, width=300, placeholder="...")
        parts.append(
            f"\nNote: This is attempt {attempt}. The previous attempt failed with:\n"
            f"{hint}\nTry a different approach."
        )
    return "\n".join(parts)


def _looks_like_failure(result: str, keywords: tuple[str, ...]) -> bool:
    lower = result.lower()
    return any(kw in lower for kw in keywords) and len(result) < 300
