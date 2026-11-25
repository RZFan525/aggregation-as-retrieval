#!/bin/bash
#SBATCH --job-name=aggmath-prompted
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# ICA Full-CoT (Prompted Aggregation) with the repo's
# scripts/run_prompted_aggregation_{4b,9b}.sh, using all of their defaults:
# K 8, MAX_MODEL_LEN 262144, MAX_TOKENS 24576, PROMPT_TOKEN_BUFFER 1024,
# thinking, temperature 1.0, top_p 0.95, 2 workers.
#   SIZE  = 2b | 4b | 9b
#   BENCH = one or more of: aime24 aime25 aime26 hmmt24 hmmt25
# The repo has no 2B wrapper, so 2B runs the 9B wrapper with only MODEL and the
# input/output/log paths changed; every other setting matches 4B/9B.
# Re-submitting resumes: finished problem_ids are skipped.
#
# Exception (agreed 2026-10-25): REQUEST_TIMEOUT=10800 instead of 300 (an
# effective 521s). Here both workers share one A100, so long generations hit the
# 521s client timeout and were silently retried, which kept shorter answers. Only
# the HTTP client timeout changes. The first run with 300 is kept in
# outputs/_backup_timeout300_20261025.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
source slurm/vllm_server.sh
resolve_array_task
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10800}"
: "${SIZE:?Set SIZE=2b|4b|9b}"
: "${BENCH:?Set BENCH, e.g. BENCH=aime24}"
print_job_info
start_gpu_watchdog

model_repo="$(qwen35_repo "${SIZE}")"
start_vllm_server "${model_repo}" 262144 "logs/slurm/vllm-prompted-${SIZE}-${SLURM_JOB_ID:-local}.log"
wait_for_vllm_server

# BENCH is intentionally unquoted so several benchmarks become separate arguments.
case "${SIZE}" in
  4b | 9b)
    # shellcheck disable=SC2086
    BASE_URL="${VLLM_BASE_URL}" bash "scripts/run_prompted_aggregation_${SIZE}.sh" ${BENCH}
    ;;
  2b)
    # shellcheck disable=SC2086
    MODEL="Qwen/Qwen3.5-2B" \
      INPUT_DIR="outputs/math_rollout/qwen3.5-2b" \
      OUTPUT_DIR="outputs/Prompted_Aggregation/qwen3.5-2b" \
      LOG_DIR="logs/prompted_aggregation_2b" \
      BASE_URL="${VLLM_BASE_URL}" \
      bash scripts/run_prompted_aggregation_9b.sh ${BENCH}
    ;;
esac

echo "done $(date -Is)"
