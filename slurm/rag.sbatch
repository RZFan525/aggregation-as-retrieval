#!/bin/bash
#SBATCH --job-name=aggmath-rag
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Single-shot RAG with scripts/run_single_shot_rag.sh. Its generation settings
# are those of ICA Full-CoT (scripts/run_prompted_aggregation_9b.sh):
# MAX_MODEL_LEN 262144, MAX_TOKENS 24576, thinking, temperature 1.0, top_p 0.95,
# 2 workers. The script evaluates each benchmark with
# baselines.prompted_aggregation.evaluate.
#   SIZE      = 2b | 4b | 9b
#   BENCH     = one or more of: aime24 aime25 aime26 hmmt24 hmmt25
#   RETRIEVER = bm25 (default) | qwen3emb (run slurm/embed.sbatch first)
#   TOP_M     = passages per problem (default 16)
# Re-submitting resumes: finished problem_ids are skipped.
#
# REQUEST_TIMEOUT=10800, the same as ICA Full-CoT in slurm/prompted.sbatch
# (agreed 2026-10-25), so a long generation is never cut off and retried.

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
start_vllm_server "${model_repo}" 262144 "logs/slurm/vllm-rag-${SIZE}-${SLURM_JOB_ID:-local}.log"
wait_for_vllm_server

# BENCH is intentionally unquoted so several benchmarks become separate arguments.
# shellcheck disable=SC2086
MODEL="${model_repo}" MODEL_SLUG="qwen3.5-${SIZE}" BASE_URL="${VLLM_BASE_URL}" \
  bash scripts/run_single_shot_rag.sh ${BENCH}

echo "done $(date -Is)"
