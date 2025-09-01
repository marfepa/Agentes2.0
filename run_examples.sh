#!/usr/bin/env bash
set -e

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -d "$DIR/venv" ]; then
  source "$DIR/venv/bin/activate"
elif [ -d "$DIR/ef-ai/venv" ]; then
  source "$DIR/ef-ai/venv/bin/activate"
else
  python3 -m venv "$DIR/venv"
  source "$DIR/venv/bin/activate"
fi

pgrep -x ollama >/dev/null || (ollama serve >/dev/null 2>&1 & sleep 1)

MODEL="mistral:7b-instruct"
ollama list | grep -q "$MODEL" || ollama pull "$MODEL"

python3 ef_chains.py
