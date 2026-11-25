#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Generation settings mirror scripts/run_prompted_aggregation_9b.sh (ICA Full-CoT);
# only the retrieved evidence differs.
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_SLUG="${MODEL_SLUG:-qwen3.5-9b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
RETRIEVER="${RETRIEVER:-bm25}"
TOP_M="${TOP_M:-16}"
INPUT_DIR="${INPUT_DIR:-outputs/math_rollout/${MODEL_SLUG}}"
INPUT_N="${INPUT_N:-8}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-outputs/SingleShot_RAG/embeddings/${MODEL_SLUG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SingleShot_RAG/${RETRIEVER}_m${TOP_M}/${MODEL_SLUG}}"
LOG_DIR="${LOG_DIR:-logs/single_shot_rag_${RETRIEVER}_m${TOP_M}_${MODEL_SLUG}}"
K="${K:-8}"
STEP_TOKEN_TARGET="${STEP_TOKEN_TARGET:-1024}"
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

echo "Running Single-shot RAG"
echo "  MODEL=${MODEL}"
echo "  BASE_URL=${BASE_URL}"
echo "  RETRIEVER=${RETRIEVER}"
echo "  TOP_M=${TOP_M}"
echo "  INPUT_DIR=${INPUT_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  LOG_DIR=${LOG_DIR}"
echo "  K=${K}"
echo "  STEP_TOKEN_TARGET=${STEP_TOKEN_TARGET}"
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

  args=(
    -m baselines.single_shot_rag.run_single_shot_rag
    --input-jsonl "${input_jsonl}"
    --retriever "${RETRIEVER}"
    --top-m "${TOP_M}"
    --k "${K}"
    --step-token-target "${STEP_TOKEN_TARGET}"
    --model "${MODEL}"
    --base-url "${BASE_URL}"
    --api-key "${API_KEY}"
    --output-dir "${OUTPUT_DIR}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-tokens "${MAX_TOKENS}"
    --prompt-token-buffer "${PROMPT_TOKEN_BUFFER}"
    --thinking "${THINKING}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-workers "${MAX_WORKERS}"
    --max-retries "${MAX_RETRIES}"
    --request-timeout "${REQUEST_TIMEOUT}"
  )
  if [[ "${RETRIEVER}" == "qwen3emb" ]]; then
    args+=(--embeddings "${EMBEDDINGS_DIR}/${benchmark}_thinking_n${INPUT_N}.npz")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi

  if agg_output="$(python "${args[@]}" 2> >(tee "${log_file}" >&2) | tail -n 1)"; then
    eval_args=(
      -m baselines.prompted_aggregation.evaluate
      --input-jsonl "${agg_output}"
      --output-dir "${OUTPUT_DIR}"
    )
    if [[ "${WRITE_PER_PROBLEM}" == "1" ]]; then
      eval_args+=(--write-per-problem)
    fi
    python "${eval_args[@]}" 2>&1 | tee -a "${log_file}"
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
