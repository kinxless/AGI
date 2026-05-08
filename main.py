"""CLI entry point.

Usage:
    python main.py "your task here"
    python main.py --no-plan "your task here"   # skip planner, run loop directly

The default path goes through the Orchestrator (Phase 3): plan → approve →
execute → memory save → report.

Pass --no-plan to bypass the planner and call the agent loop directly,
which is useful for quick single-step tasks or debugging.
"""
from __future__ import annotations

import argparse
import sys

import tools_builtin  # noqa: F401  (registers built-in tools at import time)
from agent.logger import RunLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent on a single task.")
    parser.add_argument("task", help="The task description for the agent.")
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name appended to the log file name.",
    )
    parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Skip the planner and run the agent loop directly (Phase 1/2 mode).",
    )
    args = parser.parse_args()

    logger = RunLogger(run_name=args.name)
    try:
        if args.no_plan:
            from agent.loop import AgentLoop
            loop = AgentLoop(logger=logger)
            final = loop.run(args.task)
            print("\n=== FINAL ANSWER ===")
            print(final)
        else:
            from agent.orchestrator import Orchestrator
            orch = Orchestrator(logger=logger)
            orch.run(args.task)
    finally:
        logger.close()

    print(f"\n(Full transcript: {logger.path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
