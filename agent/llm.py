"""LLM client abstraction.

Define new backends by subclassing LLMClient and implementing chat(). The
agent loop only ever talks to the base interface, so swapping Ollama for
Kimi (or any future API) is a one-line change in get_llm_client().
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

# Exported so callers can reference it without importing json directly
compact_json = json.dumps

import requests

from config import KIMI, LLM_BACKEND, OLLAMA


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a list of {role, content} messages, return the assistant text."""


class OllamaClient(LLMClient):
    def __init__(self, cfg: dict[str, Any] = OLLAMA) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.timeout = cfg["request_timeout_s"]

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


class KimiClient(LLMClient):
    """Placeholder for Phase 5. Uses an OpenAI-compatible chat completions API."""

    def __init__(self, cfg: dict[str, Any] = KIMI) -> None:
        self.base_url = cfg["base_url"].rstrip("/")
        self.model = cfg["model"]
        self.temperature = cfg["temperature"]
        self.timeout = cfg["request_timeout_s"]
        self.api_key = os.environ.get(cfg["api_key_env"], "")

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError(
                f"KimiClient requires env var {KIMI['api_key_env']} to be set."
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_llm_client() -> LLMClient:
    backend = LLM_BACKEND.lower()
    if backend == "ollama":
        return OllamaClient()
    if backend == "kimi":
        return KimiClient()
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")


def _scan_json_objects(text: str) -> list[dict[str, Any]]:
    """Return all top-level JSON objects found in text, in order."""
    results = []
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            break
        try:
            results.append(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
        pos = end + 1
    return results


def extract_json_block(text: str) -> dict[str, Any]:
    """Extract a valid agent-shaped JSON object from model output.

    Small models (3b) sometimes emit thought and action as two separate objects:
        {"thought": "..."} {"action": {...}}
    We detect this and merge adjacent objects so the loop sees one valid turn.
    """
    objects = _scan_json_objects(text)
    if not objects:
        raise ValueError("No JSON object found in model output.")

    first = objects[0]

    # Already well-formed — use as-is.
    if "action" in first or "final_answer" in first:
        return first

    # Split output: first object has only thought, second has action/final_answer.
    if "thought" in first and len(objects) >= 2:
        second = objects[1]
        if "action" in second or "final_answer" in second:
            return {**first, **second}

    return first
