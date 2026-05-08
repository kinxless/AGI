#!/usr/bin/env bash
# Run this every time the pod RESTARTS (setup_runpod.sh already ran once).
# It starts Ollama, waits for it to be ready, and prints the activate command.
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_MODEL="${OLLAMA_MODEL:-$(grep OLLAMA_MODEL "$AGENT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'qwen2.5:7b-instruct')}"

echo "[1/2] Starting Ollama..."
pkill ollama 2>/dev/null || true
sleep 2
nohup ollama serve > /tmp/ollama.log 2>&1 &

echo -n "  Waiting for Ollama"
for i in $(seq 1 30); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    sleep 2
    if [ "$i" -eq 30 ]; then
        echo ""
        echo "ERROR: Ollama failed to start. Check /tmp/ollama.log"
        exit 1
    fi
done

echo "[2/3] Loading model into VRAM..."
ollama run "$OLLAMA_MODEL" "reply with the single word: ready" --nowordwrap 2>/dev/null || true
echo "  Done."

echo "[3/3] Starting voice server..."
pkill -f 'uvicorn voice.voice_server' 2>/dev/null || true
sleep 1
cd "$AGENT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
nohup uvicorn voice.voice_server:app --host 0.0.0.0 --port 8765 \
    > /tmp/voice.log 2>&1 &
echo "  Voice server running on port 8765 (logs: /tmp/voice.log)"

echo ""
echo "Run the agent:"
echo "  cd $AGENT_DIR && source .venv/bin/activate"
echo "  python main.py \"your task here\""
