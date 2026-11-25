#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-}"
K="${K:-8}"
MODEL="${MODEL:-qwen3.5-9b}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/Prompted_Aggregation}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
PROMPT_TOKEN_BUFFER="${PROMPT_TOKEN_BUFFER:-1024}"
THINKING="${THINKING:-true}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_RETRIES="${MAX_RETRIES:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
OVERWRITE="${OVERWRITE:-0}"
WRITE_PER_PROBLEM="${WRITE_PER_PROBLEM:-0}"

usage() {
  cat <<'EOF'
Usage:
  INPUT_JSONL=outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl \
  bash scripts/run_prompted_aggregation.sh

Environment:
  INPUT_JSONL           Required grouped rollout JSONL from scripts/run_rollout.sh.
  K                     Number of rollouts to aggregate. Default: 8.
  MODEL                 Aggregator model or alias. Default: qwen3.5-9b.
  BASE_URL              OpenAI-compatible endpoint. Default: http://localhost:8000/v1.
  OUTPUT_DIR            Aggregation/eval output directory. Default: outputs/Prompted_Aggregation.
  MAX_MODEL_LEN         Served model context length. Default: 65536.
  MAX_TOKENS            Reserved generation tokens. Default: 16384.
  PROMPT_TOKEN_BUFFER   Extra reserved prompt buffer. Default: 1024.
  OVERWRITE             Set to 1 to regenerate aggregation output.
  WRITE_PER_PROBLEM     Set to 1 to write per-problem eval JSONL.
EOF
}

die() {
  echo "ERROR: $*" >&2
  echo >&2
  usage >&2
  exit 1
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "This script reads configuration from environment variables."
      ;;
  esac
fi

if [[ -z "${INPUT_JSONL}" ]]; then
  die "INPUT_JSONL is required."
fi

AGG_ARGS=(
  -m baselines.prompted_aggregation.run_prompted_aggregation
  --input-jsonl "${INPUT_JSONL}"
  --k "${K}"
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

if [[ "${OVERWRITE}" == "1" ]]; then
  AGG_ARGS+=(--overwrite)
fi

cd "${PROJECT_ROOT}"

echo "Running Prompted Aggregation with:"
echo "  INPUT_JSONL=${INPUT_JSONL}"
echo "  K=${K}"
echo "  MODEL=${MODEL}"
echo "  BASE_URL=${BASE_URL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  MAX_TOKENS=${MAX_TOKENS}"
echo "  PROMPT_TOKEN_BUFFER=${PROMPT_TOKEN_BUFFER}"
echo "  THINKING=${THINKING}"
echo "  MAX_WORKERS=${MAX_WORKERS}"

AGG_OUTPUT="$(python "${AGG_ARGS[@]}" | tail -n 1)"

EVAL_ARGS=(
  -m baselines.prompted_aggregation.evaluate
  --input-jsonl "${AGG_OUTPUT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${WRITE_PER_PROBLEM}" == "1" ]]; then
  EVAL_ARGS+=(--write-per-problem)
fi

echo "Running Prompted Aggregation evaluation:"
echo "  INPUT_JSONL=${AGG_OUTPUT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  WRITE_PER_PROBLEM=${WRITE_PER_PROBLEM}"

python "${EVAL_ARGS[@]}"
