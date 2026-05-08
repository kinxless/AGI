"""Orchestrator: top-level entry point for Phase 3.

Full flow:
  A. Search memory → inject context into planner
  B. Plan → show to user → get approval (with revision loop)
  C. Execute steps in order, replanning on failure
  D. Save summary to memory
  E. Print final report
"""
from __future__ import annotations

import sys
import textwrap
from typing import List

from config import PLANNER
from agent.llm import LLMClient, get_llm_client
from agent.logger import NullLogger, RunLogger
from agent.executor import StepExecutor, StepFailed
from agent.planner import Planner, Step

_MAX_APPROVAL_CYCLES: int = PLANNER["max_approval_cycles"]
_MAX_REPLAN_ATTEMPTS: int = PLANNER["max_replan_attempts"]
_CONTEXT_WORD_LIMIT: int = PLANNER["context_word_limit"]


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient | None = None,
        logger: RunLogger | NullLogger | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.log = logger or NullLogger()
        self.planner = Planner(llm=self.llm, logger=self.log)
        self.executor = StepExecutor(llm=self.llm, logger=self.log)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, task: str) -> None:
        print(f"\nTask received: {task}\n")

        # A — memory search
        memory_context = self._search_memory(task)

        # B — plan + approval
        steps = self._plan_and_approve(task, memory_context)
        if steps is None:
            print("\nAborted by user before execution.")
            return

        # C — execute
        completed_steps: List[Step] = []
        all_steps = list(steps)
        context = memory_context
        replan_count = 0

        i = 0
        while i < len(all_steps):
            step = all_steps[i]
            if step.status in ("done", "skipped"):
                i += 1
                continue

            print(f"\n  Executing step {step.step_number}: {step.description}")
            self.log.info(f"ORCHESTRATOR: executing step {step.step_number}")

            try:
                result = self.executor.execute_step(step, context)
                completed_steps.append(step)
                context = _update_context(context, step, result, _CONTEXT_WORD_LIMIT)
                print(f"  Step {step.step_number} complete: {textwrap.shorten(result, 120)}")
                i += 1

            except StepFailed as exc:
                self.log.info(f"ORCHESTRATOR: step {step.step_number} failed — {exc}")

                if replan_count >= _MAX_REPLAN_ATTEMPTS:
                    print(
                        f"\n  Step {step.step_number} failed and replan limit "
                        f"({_MAX_REPLAN_ATTEMPTS}) reached."
                    )
                    self._print_partial_report(task, all_steps)
                    self._save_memory(task, all_steps, partial=True)
                    return

                replan_count += 1
                new_steps = self.planner.replan(
                    task=task,
                    original_steps=all_steps,
                    failed_step=step,
                    error=str(exc),
                )

                print(f"\n{'='*60}")
                print(f"  REPLANNING (attempt {replan_count}/{_MAX_REPLAN_ATTEMPTS})")
                print(f"  Step {step.step_number} failed: {step.description}")
                print(self.planner.display_plan(new_steps))

                answer = _prompt_user("Continue with revised plan? (yes/no): ")
                self.log.info(f"ORCHESTRATOR: user replan response: {answer}")
                if answer.strip().lower() not in ("yes", "y"):
                    print("\nAborted by user after replan.")
                    self._print_partial_report(task, all_steps)
                    self._save_memory(task, all_steps, partial=True)
                    return

                # Splice in new steps, mark failed step skipped
                step.status = "skipped"
                all_steps = all_steps[: i + 1] + new_steps
                i += 1  # skip past the now-skipped failed step

        # D — save to memory
        self._save_memory(task, all_steps, partial=False)

        # E — final report
        self._print_final_report(task, all_steps)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _search_memory(self, task: str) -> str:
        from agent.tools import get_tool, ToolNotFoundError
        try:
            result = get_tool("search_memory").call({"query": task, "max_results": 5})
            if result and "No relevant memories found" not in result:
                self.log.info(f"ORCHESTRATOR: memory context retrieved:\n{result}")
                print(f"\n[Memory] Found relevant context:\n{result}\n")
                return result
        except ToolNotFoundError:
            pass
        except Exception as e:
            self.log.info(f"ORCHESTRATOR: memory search error: {e}")
        return ""

    def _save_memory(self, task: str, steps: List[Step], partial: bool) -> None:
        from agent.tools import get_tool, ToolNotFoundError
        done = [s for s in steps if s.status == "done"]
        failed = [s for s in steps if s.status == "failed"]
        summary_lines = [
            f"Task: {task}",
            f"Status: {'partial' if partial else 'completed'}",
            f"Steps done: {len(done)}/{len(steps)}",
        ]
        for s in done:
            snippet = textwrap.shorten(s.result or "", width=100, placeholder="...")
            summary_lines.append(f"  Step {s.step_number}: {s.description} → {snippet}")
        for s in failed:
            summary_lines.append(f"  FAILED step {s.step_number}: {s.description}")
        summary = "\n".join(summary_lines)
        try:
            get_tool("save_to_memory").call({"text": summary, "source": "orchestrator"})
            self.log.info(f"ORCHESTRATOR: saved to memory:\n{summary}")
            print("\n[Memory] Task summary saved to memory.")
        except ToolNotFoundError:
            pass
        except Exception as e:
            self.log.info(f"ORCHESTRATOR: save_to_memory error: {e}")

    # ------------------------------------------------------------------
    # Plan approval loop
    # ------------------------------------------------------------------

    def _plan_and_approve(
        self, task: str, memory_context: str
    ) -> List[Step] | None:
        steps = self.planner.plan(task, memory_context)
        for cycle in range(1, _MAX_APPROVAL_CYCLES + 1):
            print(self.planner.display_plan(steps))
            answer = _prompt_user("Approve this plan? (yes / no / edit): ")
            self.log.info(f"ORCHESTRATOR: plan approval cycle {cycle}, answer: {answer}")
            answer_l = answer.strip().lower()

            if answer_l in ("yes", "y"):
                return steps

            elif answer_l in ("no", "n", "edit"):
                if cycle == _MAX_APPROVAL_CYCLES:
                    print(f"\nMax revision cycles ({_MAX_APPROVAL_CYCLES}) reached. Aborting.")
                    return None
                feedback = _prompt_user("What should change? ")
                self.log.info(f"ORCHESTRATOR: user plan revision: {feedback}")
                steps = self.planner.plan(f"{task}\n\nRevise the plan: {feedback}", memory_context)

            else:
                print("  Please answer yes, no, or edit.")

        return None

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _print_final_report(self, task: str, steps: List[Step]) -> None:
        done = [s for s in steps if s.status == "done"]
        failed = [s for s in steps if s.status == "failed"]
        skipped = [s for s in steps if s.status == "skipped"]
        print(f"\n{'='*60}")
        print("  TASK COMPLETE")
        print(f"{'='*60}")
        print(f"  Task   : {task}")
        print(f"  Done   : {len(done)}  Failed: {len(failed)}  Skipped: {len(skipped)}")
        print()
        for s in steps:
            icon = {"done": "✓", "failed": "✗", "skipped": "~", "pending": "?", "running": "?"}.get(s.status, "?")
            result_line = textwrap.shorten(s.result or "", width=80, placeholder="...")
            print(f"  {icon} Step {s.step_number}: {s.description}")
            if result_line:
                print(f"       → {result_line}")
        print(f"\n{'='*60}\n")

    def _print_partial_report(self, task: str, steps: List[Step]) -> None:
        done = [s for s in steps if s.status == "done"]
        print(f"\n{'='*60}")
        print("  TASK PARTIALLY COMPLETED")
        print(f"{'='*60}")
        print(f"  Task       : {task}")
        print(f"  Completed  : {len(done)} step(s)")
        for s in done:
            print(f"    ✓ Step {s.step_number}: {s.description}")
        print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _prompt_user(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "no"


def _update_context(
    existing: str, step: Step, result: str, word_limit: int
) -> str:
    new_entry = f"Step {step.step_number} ({step.description}): {result}"
    combined = (existing + "\n" + new_entry).strip()
    words = combined.split()
    if len(words) <= word_limit:
        return combined
    # Keep the tail (most recent) within limit
    kept = words[-word_limit:]
    return "[...earlier steps summarised...] " + " ".join(kept)
