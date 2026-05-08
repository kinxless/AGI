"""Speech-to-text using faster-whisper.

Loads the model lazily on first call so importing this module is cheap.
Never raises — failures return an empty string so the WebSocket server
keeps running.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from config import WHISPER_MODEL, WHISPER_DEVICE


class SpeechToText:
    def __init__(self) -> None:
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="float16",
        )

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            self._ensure_loaded()
            # Use the correct extension so ffmpeg picks the right decoder.
            # WAV starts with "RIFF"; everything else (webm, ogg, mp4) goes
            # to ffmpeg as-is via the .webm extension — faster-whisper/pyav
            # handles any format ffmpeg can decode.
            ext = ".wav" if len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF" else ".webm"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_bytes)
                path = f.name
            try:
                segments, _info = self._model.transcribe(path)
                return "".join(s.text for s in segments).strip()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[STT] error: {e}")
            return ""


_stt: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    global _stt
    if _stt is None:
        _stt = SpeechToText()
    return _stt
