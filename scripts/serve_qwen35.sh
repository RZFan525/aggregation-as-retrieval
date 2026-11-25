#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
USE_REASONING_PARSER="${USE_REASONING_PARSER:-1}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-}"

ARGS=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
)

if [[ "${USE_REASONING_PARSER}" == "1" ]]; then
  ARGS+=(--reasoning-parser qwen3)
fi

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  ARGS+=(--language-model-only)
fi

if [[ -n "${DOWNLOAD_DIR}" ]]; then
  ARGS+=(--download-dir "${DOWNLOAD_DIR}")
fi

echo "Starting vLLM server with:"
echo "  MODEL=${MODEL}"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"
echo "  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "  DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "  DTYPE=${DTYPE}"
echo "  USE_REASONING_PARSER=${USE_REASONING_PARSER}"
echo "  ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}"
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  echo "  TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
fi
echo "  LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY}"
if [[ -n "${DOWNLOAD_DIR}" ]]; then
  echo "  DOWNLOAD_DIR=${DOWNLOAD_DIR}"
fi

if [[ "${USE_REASONING_PARSER}" == "1" ]]; then
  echo "  NOTE: If text mode fails on older vLLM builds, retry with USE_REASONING_PARSER=0."
fi

exec vllm "${ARGS[@]}"
