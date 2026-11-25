#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
INPUT_DIR="${INPUT_DIR:-outputs/math_rollout/qwen3.5-4b}"
INPUT_N="${INPUT_N:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Prompted_Aggregation/qwen3.5-4b}"
LOG_DIR="${LOG_DIR:-logs/prompted_aggregation_4b}"
K="${K:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_TOKENS="${MAX_TOKENS:-24576}"
PROMPT_TOKEN_BUFFER="${PROMPT_TOKEN_BUFFER:-1024}"
THINKING="${THINKING:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
OVERWRITE="${OVERWRITE:-0}"
WRITE_PER_PROBLEM="${WRITE_PER_PROBLEM:-0}"
CHECK_SERVER="${CHECK_SERVER:-1}"

if [[ $# -gt 0 ]]; then
  BENCHMARKS=("$@")
else
  BENCHMARKS=(aime24 aime25 aime26 hmmt24 hmmt25)
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Running Qwen3.5-4B Prompted Aggregation"
echo "  MODEL=${MODEL}"
echo "  BASE_URL=${BASE_URL}"
echo "  INPUT_DIR=${INPUT_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
echo "  K=${K}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  MAX_TOKENS=${MAX_TOKENS}"
echo "  PROMPT_TOKEN_BUFFER=${PROMPT_TOKEN_BUFFER}"
echo "  MAX_WORKERS=${MAX_WORKERS}"
echo "  REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo

if [[ "${CHECK_SERVER}" == "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    echo "Checking aggregation server: ${BASE_URL}/models"
    curl -fsS "${BASE_URL%/}/models" >/dev/null
  else
    echo "WARNING: curl is not available; skipping server check." >&2
  fi
fi

for benchmark in "${BENCHMARKS[@]}"; do
  input_jsonl="${INPUT_DIR}/${benchmark}_thinking_n${INPUT_N}.jsonl"
  if [[ ! -f "${input_jsonl}" ]]; then
    echo "ERROR: Missing input file: ${input_jsonl}" >&2
    exit 1
  fi
done

failed=()

for benchmark in "${BENCHMARKS[@]}"; do
  input_jsonl="${INPUT_DIR}/${benchmark}_thinking_n${INPUT_N}.jsonl"
  log_file="${LOG_DIR}/${benchmark}_thinking_n${INPUT_N}.log"

  echo
  echo "===== ${benchmark} started at $(date -Is) ====="
  echo "Input: ${input_jsonl}"
  echo "Log: ${log_file}"

  if INPUT_JSONL="${input_jsonl}" \
    MODEL="${MODEL}" \
    BASE_URL="${BASE_URL}" \
    API_KEY="${API_KEY}" \
    K="${K}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    PROMPT_TOKEN_BUFFER="${PROMPT_TOKEN_BUFFER}" \
    THINKING="${THINKING}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    MAX_WORKERS="${MAX_WORKERS}" \
    MAX_RETRIES="${MAX_RETRIES}" \
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    OVERWRITE="${OVERWRITE}" \
    WRITE_PER_PROBLEM="${WRITE_PER_PROBLEM}" \
    bash scripts/run_prompted_aggregation.sh 2>&1 | tee "${log_file}"; then
    echo "===== ${benchmark} finished at $(date -Is) ====="
  else
    echo "===== ${benchmark} failed at $(date -Is) =====" >&2
    failed+=("${benchmark}")
  fi
done

echo
echo "Evaluation summaries in ${OUTPUT_DIR}:"
find "${OUTPUT_DIR}" -maxdepth 1 -name "*.eval.json" -print | sort

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed benchmarks: ${failed[*]}" >&2
  exit 1
fi

echo "All requested benchmarks completed."
