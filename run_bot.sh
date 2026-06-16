#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

MODEL="qwen3:4b"

if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Запускаю Ollama-сервер..."
    ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do
        curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 0.5
    done
fi

if ! ollama list | grep -q "$MODEL"; then
    echo "Скачиваю модель $MODEL (один раз)..."
    ollama pull "$MODEL"
fi

echo "Запускаю бота..."
exec .venv/bin/python scripts/rag_bot.py
