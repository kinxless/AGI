"""FastAPI WebSocket voice server.

Pipeline per audio message:
    audio bytes (base64)
        -> STT (faster-whisper)
        -> Orchestrator.run(text)         (Phase 3 stack, full plan/execute)
        -> TTS (kokoro-onnx)
        -> {text, audio (base64)}

The Orchestrator is interactive (it asks the user to approve plans on
stdin and prints to stdout). For voice we redirect stdin to feed "yes"
automatically, and tee stdout so we can capture the agent's final answer
while still showing it on the server console.

Single-client only. Each step is sent as its own WebSocket message so
the laptop client can show real-time status. All exceptions are sent
back as {"type": "error", "message": "..."} — the server never crashes.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from typing import Any

import tools_builtin  # noqa: F401  (registers built-in tools at import time)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice.stt import get_stt
from voice.tts import get_tts


app = FastAPI()

# A single lock around the agent run so concurrent /voice connections
# can't trample each other's stdin/stdout redirection.
_agent_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "stt": True, "tts": True}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/voice")
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
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        await _send_error(ws, f"Bad audio payload: {e}")
        return

    loop = asyncio.get_running_loop()

    # 2. Transcribe
    print("[voice] transcribing audio...")
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

    # 3. Run agent
    await ws.send_json({"type": "thinking"})
    print(f"[voice] running agent on: {text!r}")
    try:
        async with _agent_lock:
            answer = await loop.run_in_executor(None, _run_orchestrator, text)
    except Exception as e:
        await _send_error(ws, f"Agent error: {e}")
        return

    answer = (answer or "").strip()
    if not answer:
        answer = "The agent finished but produced no spoken output."
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
# Orchestrator wrapper (sync — runs in a thread pool from the WS handler)
# ---------------------------------------------------------------------------

def _run_orchestrator(task: str) -> str:
    """Run the full Orchestrator and capture its final answer.

    The Orchestrator is interactive (input() for plan approval) and prints
    everything to stdout. We:
      * feed 'yes' to stdin so plans are auto-approved
      * tee stdout so the captured text contains the agent loop's DONE
        block (the actual final answer) while still letting us see logs
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
    """Write to multiple streams; tolerate per-stream errors."""

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

    The agent loop prints a "DONE" block containing the step's full
    final answer; the *last* DONE block belongs to the final step.
    Falls back to the TASK COMPLETE summary, then to raw tail text.
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
