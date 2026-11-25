#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SUMMARY_MODEL="${SUMMARY_MODEL:-${MODEL}}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
SUMMARY_BASE_URL="${SUMMARY_BASE_URL:-${BASE_URL}}"
API_KEY="${API_KEY:-EMPTY}"
SUMMARY_API_KEY="${SUMMARY_API_KEY:-${API_KEY}}"
INPUT_DIR="${INPUT_DIR:-outputs/math_rollout/qwen3.5-4b}"
INPUT_N="${INPUT_N:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Aggent_Aggregation}"
LOG_DIR="${LOG_DIR:-logs/agentic_aggregation_4b}"
K="${K:-8}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-262144}"
SOLUTION_STEP_TOKENS="${SOLUTION_STEP_TOKENS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-4096}"
THINKING="${THINKING:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
SEARCH_TOOL="${SEARCH_TOOL:-rouge}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
WRITE_PER_PROBLEM="${WRITE_PER_PROBLEM:-0}"
CHECK_SERVER="${CHECK_SERVER:-1}"
MODEL_SLUG="${MODEL_SLUG:-}"

if [[ $# -gt 0 ]]; then
  BENCHMARKS=("$@")
else
  BENCHMARKS=(aime24 aime25 aime26 hmmt24 hmmt25)
fi

cd "${PROJECT_ROOT}"
if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="$(
    MODEL="${MODEL}" python -c 'import os, re; from aggmath.rollout.qwen_api import normalize_model_name; model = normalize_model_name(os.environ["MODEL"]); name = model.strip().split("/")[-1].lower(); print(re.sub(r"[^a-z0-9._-]+", "_", name).strip("_") or "item")'
  )"
fi
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_SLUG}"
mkdir -p "${MODEL_OUTPUT_DIR}" "${LOG_DIR}"

echo "Running Qwen3.5-4B Agentic Aggregation"
echo "  MODEL=${MODEL}"
echo "  SUMMARY_MODEL=${SUMMARY_MODEL}"
echo "  BASE_URL=${BASE_URL}"
echo "  SUMMARY_BASE_URL=${SUMMARY_BASE_URL}"
echo "  INPUT_DIR=${INPUT_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MODEL_OUTPUT_DIR=${MODEL_OUTPUT_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
echo "  K=${K}"
echo "  MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
echo "  SOLUTION_STEP_TOKENS=${SOLUTION_STEP_TOKENS}"
echo "  MAX_TOKENS=${MAX_TOKENS}"
echo "  SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS}"
echo "  THINKING=${THINKING}"
echo "  MAX_WORKERS=${MAX_WORKERS}"
echo "  REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "  SEARCH_TOOL=${SEARCH_TOOL}"
echo

if [[ "${CHECK_SERVER}" == "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    echo "Checking aggregation server: ${BASE_URL}/models"
    curl -fsS "${BASE_URL%/}/models" >/dev/null
    if [[ "${SUMMARY_BASE_URL}" != "${BASE_URL}" ]]; then
      echo "Checking summary server: ${SUMMARY_BASE_URL}/models"
      curl -fsS "${SUMMARY_BASE_URL%/}/models" >/dev/null
    fi
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
  output_jsonl="${MODEL_OUTPUT_DIR}/${benchmark}_agentic_n${K}.jsonl"
  eval_json="${MODEL_OUTPUT_DIR}/${benchmark}_agentic_n${K}.eval.json"

  echo
  echo "===== ${benchmark} started at $(date -Is) ====="
  echo "Input: ${input_jsonl}"
  echo "Log: ${log_file}"

  args=(
    -m aggmath.aggregation.run_agentic_aggregation
    --input-jsonl "${input_jsonl}"
    --model "${MODEL}"
    --base-url "${BASE_URL}"
    --api-key "${API_KEY}"
    --summary-model "${SUMMARY_MODEL}"
    --summary-base-url "${SUMMARY_BASE_URL}"
    --summary-api-key "${SUMMARY_API_KEY}"
    --k "${K}"
    --max-context-tokens "${MAX_CONTEXT_TOKENS}"
    --solution-step-tokens "${SOLUTION_STEP_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --summary-max-tokens "${SUMMARY_MAX_TOKENS}"
    --thinking "${THINKING}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-workers "${MAX_WORKERS}"
    --max-retries "${MAX_RETRIES}"
    --request-timeout "${REQUEST_TIMEOUT}"
    --search-tool "${SEARCH_TOOL}"
    --output-dir "${OUTPUT_DIR}"
  )

  if [[ -n "${TOKENIZER_NAME}" ]]; then
    args+=(--tokenizer-name "${TOKENIZER_NAME}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if python "${args[@]}" 2>&1 | tee "${log_file}"; then
    if [[ "${SKIP_EVAL}" != "1" ]]; then
      echo "Running evaluation: ${output_jsonl}" | tee -a "${log_file}"
      eval_args=(
        -m aggmath.aggregation.evaluation
        --input-jsonl "${output_jsonl}"
      )
      if [[ "${WRITE_PER_PROBLEM}" == "1" ]]; then
        eval_args+=(--write-per-problem)
      fi
      python "${eval_args[@]}" 2>&1 | tee -a "${log_file}"
      echo "Evaluation result JSON: ${eval_json}" | tee -a "${log_file}"
      python -m json.tool "${eval_json}" | tee -a "${log_file}"
    fi
    echo "===== ${benchmark} finished at $(date -Is) ====="
  else
    echo "===== ${benchmark} failed at $(date -Is) =====" >&2
    failed+=("${benchmark}")
  fi
done

echo
echo "Aggregation outputs in ${MODEL_OUTPUT_DIR}:"
find "${MODEL_OUTPUT_DIR}" -maxdepth 1 -name "*.jsonl" -print | sort
echo
echo "Evaluation summaries in ${MODEL_OUTPUT_DIR}:"
find "${MODEL_OUTPUT_DIR}" -maxdepth 1 -name "*.eval.json" -print | sort

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed benchmarks: ${failed[*]}" >&2
  exit 1
fi

echo "All requested benchmarks completed."
