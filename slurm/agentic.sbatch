#!/bin/bash
#SBATCH --job-name=aggmath-agentic
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Agentic aggregation with the repo's scripts/run_agentic_aggregation_{2b,4b,9b}.sh,
# using all of their defaults: K 8, MAX_CONTEXT_TOKENS 262144,
# SOLUTION_STEP_TOKENS 1024, MAX_TOKENS 16384, SUMMARY_MAX_TOKENS 4096, thinking,
# temperature 1.0, top_p 0.95, rouge search tool, summary model = same model
# and server, 2 workers. The script runs aggmath.aggregation.evaluation after
# each benchmark.
#   SIZE  = 2b | 4b | 9b
#   BENCH = one or more of: aime24 aime25 aime26 hmmt24 hmmt25
# Re-submitting resumes: finished problem_ids are skipped.
#
# Exception (agreed 2026-10-25): REQUEST_TIMEOUT=10800 instead of 300 (it is
# LiteLLM's per-call timeout). With both workers sharing one A100, a long agent
# turn can pass 300s and get silently retried. Only the client timeout changes.
# The first run with 300 is kept in outputs/_backup_timeout300_20261025.

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
# The agent's context budget (MAX_CONTEXT_TOKENS) is 262144, so serve that length.
start_vllm_server "${model_repo}" 262144 "logs/slurm/vllm-agentic-${SIZE}-${SLURM_JOB_ID:-local}.log"
wait_for_vllm_server

# BENCH is intentionally unquoted so several benchmarks become separate arguments.
# shellcheck disable=SC2086
BASE_URL="${VLLM_BASE_URL}" bash "scripts/run_agentic_aggregation_${SIZE}.sh" ${BENCH}

echo "done $(date -Is)"
