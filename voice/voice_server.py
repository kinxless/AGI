"""FastAPI voice server: Jarvis web UI + STT/agent/TTS pipeline.

Endpoints
---------
GET  /          -> templates/index.html (the Jarvis UI)
GET  /health    -> {"status": "ok"}
GET  /static/*  -> static assets (currently empty placeholder dir)
WS   /ws        -> audio in / agent run / audio out

WebSocket protocol
------------------
Client -> server:
    {"type": "ping"}
    {"type": "audio", "data": "<base64-encoded WAV bytes>"}

Server -> client:
    {"type": "pong"}
    {"type": "transcribed", "text": "..."}
    {"type": "thinking"}
    {"type": "response",   "text": "...", "audio": "<base64 WAV>"}
    {"type": "error",      "message": "..."}

Design notes
------------
* The Orchestrator is interactive (input() for plan approval, prints to
  stdout). For voice we redirect stdin to feed "yes" automatically and
  tee stdout so we can extract the agent loop's DONE block as the final
  spoken answer. Phase 1-4 code is not modified.
* All blocking work (STT, agent, TTS) runs in a thread executor so the
  WebSocket event loop never blocks.
* If the browser sends webm/opus instead of WAV, _ensure_wav() runs it
  through pydub (ffmpeg) as a fallback before STT.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import tools_builtin  # noqa: F401  (registers built-in tools at import time)
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from voice.stt import get_stt
from voice.tts import get_tts


_VOICE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _VOICE_DIR / "templates"
_STATIC_DIR = _VOICE_DIR / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Single agent run at a time — protects stdin/stdout redirection.
_agent_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    print("[voice] client connected")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await _send_error(ws, f"Invalid JSON: {e}")
                continue
            await _handle_message(ws, msg)
    except WebSocketDisconnect:
        print("[voice] client disconnected")
    except Exception as e:
        print(f"[voice] websocket error: {e}")
        await _send_error(ws, f"Server error: {e}")


async def _handle_message(ws: WebSocket, msg: dict[str, Any]) -> None:
    msg_type = msg.get("type")
    if msg_type == "ping":
        await ws.send_json({"type": "pong"})
        return
    if msg_type == "audio":
        await _handle_audio(ws, msg)
        return
    await _send_error(ws, f"Unknown message type: {msg_type}")


async def _handle_audio(ws: WebSocket, msg: dict[str, Any]) -> None:
    # 1. Decode
    try:
        audio_b64 = msg.get("data") or ""
        raw = base64.b64decode(audio_b64)
    except Exception as e:
        await _send_error(ws, f"Bad audio payload: {e}")
        return

    audio_bytes = _ensure_wav(raw)
    loop = asyncio.get_running_loop()

    # 2. Transcribe
    print(f"[voice] transcribing {len(audio_bytes):,} bytes...")
    try:
        text = await loop.run_in_executor(None, get_stt().transcribe, audio_bytes)
    except Exception as e:
        await _send_error(ws, f"STT failed: {e}")
        return

    text = (text or "").strip()
    if len(text) < 3:
        await _send_error(ws, "Could not transcribe audio")
        return

    print(f"[voice] transcribed: {text!r}")
    await ws.send_json({"type": "transcribed", "text": text})

    # 3. Run agent (synchronous — runs in executor)
    await ws.send_json({"type": "thinking"})
    print(f"[voice] running agent on: {text!r}")
    try:
        async with _agent_lock:
            answer = await loop.run_in_executor(None, _run_orchestrator, text)
    except Exception as e:
        await _send_error(ws, f"Agent error: {e}")
        return

    answer = (answer or "").strip() or "The agent finished but produced no spoken output."
    print(f"[voice] agent answer ({len(answer)} chars)")

    # 4. Synthesize
    print("[voice] synthesizing speech...")
    try:
        audio_out = await loop.run_in_executor(None, get_tts().synthesize, answer)
    except Exception as e:
        await _send_error(ws, f"TTS failed: {e}")
        return

    # 5. Respond
    await ws.send_json({
        "type": "response",
        "text": answer,
        "audio": base64.b64encode(audio_out).decode("ascii"),
    })
    print("[voice] response sent")


async def _send_error(ws: WebSocket, message: str) -> None:
    print(f"[voice] error: {message}")
    try:
        await ws.send_json({"type": "error", "message": message})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_wav(audio_bytes: bytes) -> bytes:
    """Return WAV bytes. Convert webm/opus/etc. via pydub if needed.

    Falls back to passing the raw bytes through if pydub/ffmpeg aren't
    available — faster-whisper can decode many formats via pyav anyway.
    """
    if len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF":
        return audio_bytes
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        out = io.BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception as e:
        print(f"[voice] _ensure_wav passthrough: {e}")
        return audio_bytes


def _run_orchestrator(task: str) -> str:
    """Run the full Orchestrator and capture its final answer.

    The Orchestrator prints to stdout and reads input() for plan
    approval. We redirect stdin to feed "yes" automatically, and tee
    stdout so the captured text contains the agent loop's DONE block.
    """
    from agent.orchestrator import Orchestrator
    from agent.logger import RunLogger

    captured = io.StringIO()

    real_stdin = sys.stdin
    real_stdout = sys.stdout
    sys.stdin = io.StringIO("yes\n" * 50)
    sys.stdout = _Tee(real_stdout, captured)

    logger = RunLogger(run_name="voice")
    try:
        Orchestrator(logger=logger).run(task)
    finally:
        sys.stdin = real_stdin
        sys.stdout = real_stdout
        logger.close()

    return _extract_answer(captured.getvalue())


class _Tee:
    """Writeable that forwards to multiple streams; tolerates per-stream errors."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            try:
                stream.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _extract_answer(captured: str) -> str:
    """Pull a TTS-friendly answer from captured orchestrator output.

    Strategy: the agent loop prints a "DONE" block containing each step's
    full final answer. The *last* DONE block is the final step's answer.
    Falls back to TASK COMPLETE summary, then to raw tail text.
    """
    last_done = captured.rfind("DONE")
    if last_done != -1:
        block = captured[last_done:]
        body_lines: list[str] = []
        in_body = False
        for line in block.splitlines():
            if line.startswith("="):
                if in_body:
                    break
                continue
            if line.startswith("---"):
                in_body = True
                continue
            if in_body:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if body:
            return body

    idx = captured.find("TASK COMPLETE")
    if idx != -1:
        return captured[idx : idx + 2000].strip()

    return captured[-1500:].strip()
