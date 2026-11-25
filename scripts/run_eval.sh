#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================
# Edit this block, then run:
#   bash scripts/run_eval.sh
# =========================

# Space-separated or newline-separated files / directories / globs.
# Examples:
#   INPUTS="outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl"
#   INPUTS="outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl outputs/math_rollout/qwen3.5-9b/hmmt25_thinking_n8.jsonl"
#   INPUTS="outputs/math_rollout/qwen3.5-9b"
#   INPUTS="outputs/math_rollout/qwen3.5-9b/*.jsonl"
INPUTS="outputs/math_rollout/qwen3.5-2b/aime24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-2b/aime25_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-2b/aime26_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-2b/hmmt24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-2b/hmmt25_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-4b/aime24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-4b/aime25_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-4b/aime26_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-4b/hmmt24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-4b/hmmt25_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-9b/aime25_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-9b/aime26_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-9b/hmmt24_thinking_n8.jsonl
outputs/math_rollout/qwen3.5-9b/hmmt25_thinking_n8.jsonl"

# Optional array form. Use this instead of INPUTS if any path contains spaces.
# INPUT_LIST takes priority over INPUTS when it is non-empty.
INPUT_LIST=(
  # "outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl"
  # "outputs/math_rollout/qwen3.5-9b/hmmt25_thinking_n8.jsonl"
)

# Space-separated list of k values for pass@k.
K="${K:-1 8}"

# Directory for per-file .eval.json outputs. Leave empty to write next to inputs.
OUTPUT_DIR="outputs/eval"

# Optional combined summary JSON path. Leave empty to skip.
SUMMARY_OUT="${SUMMARY_OUT:-}"

# Set to 1 to also write per-problem .eval.jsonl files.
WRITE_PER_PROBLEM="${WRITE_PER_PROBLEM:-0}"

usage() {
  cat <<'EOF'
Usage:
  Edit the config block near the top of scripts/run_eval.sh, then run:
    bash scripts/run_eval.sh

Main fields:
  INPUTS              Space-separated or newline-separated files / directories / globs.
  INPUT_LIST          Array form for multiple paths or paths containing spaces.
  OUTPUT_DIR          Directory for per-file .eval.json outputs.
  K                   Space-separated pass@k values.
  SUMMARY_OUT         Optional combined summary JSON path.
  WRITE_PER_PROBLEM   Set to 1 to write per-problem .eval.jsonl files.

Environment variables with the same names are still supported.
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
      die "This script reads inputs from its config block; edit INPUTS/OUTPUT_DIR in scripts/run_eval.sh instead of passing command-line paths."
      ;;
  esac
fi

INPUT_ARGS=()
if [[ ${#INPUT_LIST[@]} -gt 0 ]]; then
  INPUT_ARGS=("${INPUT_LIST[@]}")
elif [[ -n "${INPUTS}" ]]; then
  INPUTS_ONE_LINE="${INPUTS//$'\n'/ }"
  read -r -a INPUT_ARGS <<< "${INPUTS_ONE_LINE}"
fi

K_ARGS=()
if [[ -n "${K}" ]]; then
  read -r -a K_ARGS <<< "${K}"
fi

if [[ ${#INPUT_ARGS[@]} -eq 0 ]]; then
  die "INPUTS is required. Set INPUTS=\"file1.jsonl file2.jsonl ...\" in scripts/run_eval.sh."
fi

if [[ ${#K_ARGS[@]} -eq 0 ]]; then
  die "K is required. Set K=\"1 8\" in scripts/run_eval.sh."
fi

ARGS=(
  -m aggmath.evaluation.run_eval
  --inputs "${INPUT_ARGS[@]}"
  --k "${K_ARGS[@]}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${SUMMARY_OUT}" ]]; then
  ARGS+=(--summary-out "${SUMMARY_OUT}")
fi

if [[ "${WRITE_PER_PROBLEM}" == "1" ]]; then
  ARGS+=(--write-per-problem)
fi

cd "${PROJECT_ROOT}"

echo "Running evaluation with:"
echo "  INPUTS=${INPUT_ARGS[*]}"
echo "  K=${K_ARGS[*]}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "  OUTPUT_DIR=${OUTPUT_DIR}"
fi
if [[ -n "${SUMMARY_OUT}" ]]; then
  echo "  SUMMARY_OUT=${SUMMARY_OUT}"
fi
echo "  WRITE_PER_PROBLEM=${WRITE_PER_PROBLEM}"

python "${ARGS[@]}"
