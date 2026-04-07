#!/bin/bash
# Start Jarvis via GoClaw on Jetson Orin Nano
# Single model: Qwen3.5-2B (conversation + tools) + GoClaw gateway
#
# Usage: ./start-jarvis.sh

set -e

LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
QWEN_MODEL="$HOME/models/Qwen3.5-2B-Q4_K_M.gguf"
GOCLAW_BIN="$HOME/goclaw/goclaw-arm64"
GOCLAW_CONFIG="$HOME/goclaw/config.jarvis.json"

QWEN_PORT=8081

export LD_LIBRARY_PATH="$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH"
export CUDA_SCALE_LAUNCH_QUEUES=1

PIDS=()
cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null
    done
    echo "All stopped."
}
trap cleanup EXIT INT TERM

# --- Start Qwen3.5-2B (GPU, with KV cache quantization) ---
if ! curl -s "http://127.0.0.1:$QWEN_PORT/health" > /dev/null 2>&1; then
    echo "Starting Qwen3.5-2B on port $QWEN_PORT..."
    $LLAMA_SERVER \
        -m "$QWEN_MODEL" \
        --host 127.0.0.1 --port $QWEN_PORT \
        -ngl 999 -c 8192 -np 1 --threads 4 \
        -ctk q4_0 -ctv q4_0 --flash-attn on \
        --reasoning-budget 0 \
        --chat-template-kwargs '{"enable_thinking": false}' \
        > /tmp/llama-qwen.log 2>&1 &
    PIDS+=($!)
    echo -n "  Waiting"
    for i in $(seq 1 60); do
        curl -s "http://127.0.0.1:$QWEN_PORT/health" > /dev/null 2>&1 && echo " ready!" && break
        if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then echo " CRASHED"; tail -5 /tmp/llama-qwen.log; exit 1; fi
        echo -n "."; sleep 1
    done
    if ! curl -s "http://127.0.0.1:$QWEN_PORT/health" > /dev/null 2>&1; then
        echo " ERROR: Qwen server did not become healthy"
        tail -10 /tmp/llama-qwen.log
        exit 1
    fi
else
    echo "Qwen3.5-2B already running on port $QWEN_PORT"
fi

# --- Start GoClaw gateway ---
echo "Starting GoClaw gateway..."
exec "$GOCLAW_BIN" --config "$GOCLAW_CONFIG"
