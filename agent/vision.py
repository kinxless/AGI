"""VisionAnalyzer: send a screenshot to the Ollama vision model for analysis.

Uses Ollama's /api/generate endpoint with the images field — compatible with
any multimodal model available locally (qwen2.5vl:7b, llava, etc.).
"""
from __future__ import annotations

import base64
from typing import Optional

import requests

from config import OLLAMA, VISION_MODEL

_VISION_TIMEOUT = 60


class VisionAnalyzer:
    def __init__(self) -> None:
        self.base_url = OLLAMA["base_url"].rstrip("/")
        self.model = VISION_MODEL

    def analyze(self, image_bytes: bytes, question: str) -> str:
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "model": self.model,
                "prompt": question,
                "images": [b64],
                "stream": False,
            }
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=_VISION_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"Vision error: {e}"


_vision_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer
