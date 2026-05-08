#!/usr/bin/env bash
# Setup script for a RunPod GPU pod (Ubuntu).
# Run once after the pod starts:  bash setup_runpod.sh
set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
RAG_REPO_URL="${RAG_REPO_URL:-}"          # set to your RAG repo git URL if needed
RAG_REPO_DIR="${RAG_REPO_PATH:-/workspace/rag_system}"

echo "=== [1/5] Installing system deps ==="
apt-get update -qq && apt-get install -y -qq curl git python3-venv python3-pip

echo "=== [2/5] Installing Ollama ==="
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
fi

echo "=== [3/5] Starting Ollama and pulling model: $OLLAMA_MODEL ==="
ollama serve &>/tmp/ollama.log &
sleep 5
ollama pull "$OLLAMA_MODEL"
echo "Ollama ready."

echo "=== [4/5] Python virtual environment ==="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Install RAG repo dependencies if the repo exists or a URL was provided
if [ -n "$RAG_REPO_URL" ] && [ ! -d "$RAG_REPO_DIR" ]; then
    echo "Cloning RAG repo from $RAG_REPO_URL..."
    git clone "$RAG_REPO_URL" "$RAG_REPO_DIR"
fi
if [ -f "$RAG_REPO_DIR/requirements.txt" ]; then
    echo "Installing RAG repo dependencies..."
    pip install --quiet -r "$RAG_REPO_DIR/requirements.txt"
fi

echo "=== [5/5] Writing .env ==="
if [ ! -f .env ]; then
    cp .env.example .env
    # Patch RAG path to Linux location
    sed -i "s|^# RAG_REPO_PATH=.*|RAG_REPO_PATH=$RAG_REPO_DIR|" .env
    sed -i "s|^# RAG_REPO_PATH=/workspace.*|RAG_REPO_PATH=$RAG_REPO_DIR|" .env
    echo "RAG_REPO_PATH=$RAG_REPO_DIR" >> .env
    echo ".env created. Edit it if needed before running the agent."
else
    echo ".env already exists, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate venv : source .venv/bin/activate"
echo "Run agent     : python main.py \"your task here\""
