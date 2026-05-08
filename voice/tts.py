"""Text-to-speech using kokoro-onnx.

Loads the model lazily on first call. Never raises — failures return
empty bytes so the WebSocket server keeps running.

The kokoro-onnx package needs two model files:
  - KOKORO_MODEL_PATH    (e.g. kokoro-v1.0.onnx)
  - KOKORO_VOICES_PATH   (e.g. voices-v1.0.bin)
Both default to the current working directory; override via env vars.
"""
from __future__ import annotations

import io
from typing import Optional

from config import TTS_VOICE, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH


class TextToSpeech:
    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from kokoro_onnx import Kokoro
        self._model = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

    def synthesize(self, text: str) -> bytes:
        try:
            self._ensure_loaded()
            samples, sample_rate = self._model.create(
                text,
                voice=TTS_VOICE,
                speed=1.0,
                lang="en-us",
            )
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, samples, sample_rate, format="WAV")
            return buf.getvalue()
        except Exception as e:
            print(f"[TTS] error: {e}")
            return b""


_tts: Optional[TextToSpeech] = None


def get_tts() -> TextToSpeech:
    global _tts
    if _tts is None:
        _tts = TextToSpeech()
    return _tts
