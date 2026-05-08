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
            ext = ".wav" if len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF" else ".webm"

            # Save a copy for inspection — `scp` it back to listen yourself.
            try:
                import time as _time
                os.makedirs("/tmp/voice_debug", exist_ok=True)
                debug_path = f"/tmp/voice_debug/audio_{int(_time.time()*1000)}{ext}"
                with open(debug_path, "wb") as df:
                    df.write(audio_bytes)
                print(f"[STT] saved debug audio: {debug_path} ({len(audio_bytes):,} bytes)")
            except Exception as _e:
                print(f"[STT] debug save failed: {_e}")

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(audio_bytes)
                path = f.name
            try:
                segments, info = self._model.transcribe(
                    path,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    no_speech_threshold=0.6,
                )
                print(
                    f"[STT] decoded: duration={info.duration:.2f}s "
                    f"lang={info.language} prob={info.language_probability:.2f}"
                )
                segs = list(segments)
                for s in segs:
                    print(
                        f"[STT] seg [{s.start:.2f}-{s.end:.2f}] "
                        f"no_speech={s.no_speech_prob:.2f} text={s.text!r}"
                    )
                text = "".join(s.text for s in segs).strip()
                print(f"[STT] raw text: {text!r}")
                if text.lower().strip(".! ") in {"you", "thank you", "thanks"}:
                    print(f"[STT] discarding hallucination: {text!r}")
                    return ""
                return text
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
