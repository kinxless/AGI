"""Self-contained voice server test.

Runs entirely on the pod — no laptop needed.

Steps:
  1. Health check (HTTP GET /health)
  2. Ping (WebSocket)
  3. Full round-trip: synthesize a spoken question with TTS, send it to
     the voice server as audio, print every message back from the server.

Usage (voice server must already be running on port 8765):
    python test_voice.py
    python test_voice.py --question "What is the capital of France?"
    python test_voice.py --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--question",
        default="What is two plus two?",
        help="Question to speak and send to the agent.",
    )
    p.add_argument(
        "--wav",
        default=None,
        help="Path to an existing WAV file to send instead of synthesizing one.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> bool:
    print(f"\n[1/3] Health check → GET {base_url}/health")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        data = r.json()
        print(f"      {data}")
        return data.get("status") == "ok"
    except Exception as e:
        print(f"      FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# 2. Synthesize question → WAV bytes
# ---------------------------------------------------------------------------

def synthesize_question(question: str) -> bytes:
    print(f"\n[2/3] Synthesizing question with TTS: {question!r}")
    try:
        from voice.tts import get_tts
        wav = get_tts().synthesize(question)
        if not wav:
            print("      TTS returned empty bytes — check kokoro model files.")
            sys.exit(1)
        print(f"      OK — {len(wav):,} bytes of WAV audio")
        return wav
    except Exception as e:
        print(f"      FAILED: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 3. Full round-trip WebSocket test
# ---------------------------------------------------------------------------

async def run_ws_test(ws_url: str, wav_bytes: bytes) -> None:
    import websockets

    print(f"\n[3/3] WebSocket test → {ws_url}")

    async with websockets.connect(ws_url) as ws:
        # --- ping ---
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        assert pong.get("type") == "pong", f"Expected pong, got: {pong}"
        print("      ping → pong ✓")

        # --- audio message ---
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        await ws.send(json.dumps({"type": "audio", "data": audio_b64}))
        print("      audio sent — waiting for responses...")

        # Receive until we get a 'response' or 'error'
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=300)
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "transcribed":
                print(f"\n  [transcribed] {msg['text']!r}")

            elif msg_type == "thinking":
                print("  [thinking]   agent is working...")

            elif msg_type == "response":
                print(f"\n  [response]")
                print(f"    text  : {msg['text']}")
                audio_out = base64.b64decode(msg.get("audio", ""))
                print(f"    audio : {len(audio_out):,} bytes WAV")
                # Save so you can scp it back to listen
                out_path = "/tmp/voice_response.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_out)
                print(f"    saved : {out_path}  (scp to listen)")
                break

            elif msg_type == "error":
                print(f"\n  [error] {msg.get('message')}")
                break

            else:
                print(f"  [unknown] {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/ws"

    # 1. Health
    if not check_health(base_url):
        print("\nServer not healthy — is `uvicorn voice.voice_server:app ...` running?")
        sys.exit(1)

    # 2. Audio
    if args.wav:
        with open(args.wav, "rb") as f:
            wav_bytes = f.read()
        print(f"\n[2/3] Using existing WAV: {args.wav} ({len(wav_bytes):,} bytes)")
    else:
        wav_bytes = synthesize_question(args.question)

    # 3. Round-trip
    asyncio.run(run_ws_test(ws_url, wav_bytes))


if __name__ == "__main__":
    main()
