#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-qwen3.5-9b}"
BENCHMARKS="${BENCHMARKS:-aime24 aime25 aime26 hmmt24 hmmt25}"
THINKING="${THINKING:-true}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/math_rollout}"
ROLLOUT_COUNT="${ROLLOUT_COUNT:-8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
LONGCOT_SPLIT="${LONGCOT_SPLIT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
EXTRA_INSTRUCTION="${EXTRA_INSTRUCTION:-}"
SEED="${SEED:-}"
OVERWRITE="${OVERWRITE:-0}"

ARGS=(
  -m aggmath.rollout.run_math_rollout
  --model "${MODEL}"
  --benchmarks ${BENCHMARKS}
  --thinking "${THINKING}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --output-dir "${OUTPUT_DIR}"
  --rollout-count "${ROLLOUT_COUNT}"
  --max-tokens "${MAX_TOKENS}"
  --max-workers "${MAX_WORKERS}"
  --max-retries "${MAX_RETRIES}"
  --request-timeout "${REQUEST_TIMEOUT}"
)

if [[ -n "${LONGCOT_SPLIT}" ]]; then
  ARGS+=(--longcot-split "${LONGCOT_SPLIT}")
fi

if [[ -n "${DATASET_ROOT}" ]]; then
  ARGS+=(--dataset-root "${DATASET_ROOT}")
fi

if [[ -n "${EXTRA_INSTRUCTION}" ]]; then
  ARGS+=(--extra-instruction "${EXTRA_INSTRUCTION}")
fi

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed "${SEED}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "${PROJECT_ROOT}"

echo "Running rollout with:"
echo "  MODEL=${MODEL}"
echo "  BENCHMARKS=${BENCHMARKS}"
echo "  THINKING=${THINKING}"
echo "  BASE_URL=${BASE_URL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  ROLLOUT_COUNT=${ROLLOUT_COUNT}"
echo "  MAX_TOKENS=${MAX_TOKENS}"
echo "  MAX_WORKERS=${MAX_WORKERS}"
echo "  MAX_RETRIES=${MAX_RETRIES}"
echo "  REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
if [[ -n "${LONGCOT_SPLIT}" ]]; then
  echo "  LONGCOT_SPLIT=${LONGCOT_SPLIT}"
fi
if [[ -n "${DATASET_ROOT}" ]]; then
  echo "  DATASET_ROOT=${DATASET_ROOT}"
fi
if [[ -n "${SEED}" ]]; then
  echo "  SEED=${SEED}"
fi

python "${ARGS[@]}"
