#!/bin/bash
#SBATCH --job-name=aggmath-rollout
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Generate the 8 candidate solutions with the repo's scripts/run_rollout.sh.
#   SIZE  = 2b | 4b | 9b
#   BENCH = one or more of: aime24 aime25 aime26 hmmt24 hmmt25
# Only the server endpoint and the benchmark list are set here. All generation
# parameters are the script defaults: thinking, n=8, max_tokens 32768,
# temperature 1.0, top_p 0.95, top_k 20, 16 workers, no seed.
# Re-submitting resumes: problem_ids already in the output JSONL are skipped.
#
# The one exception is REQUEST_TIMEOUT. Here a single A100 serves all
# 16 workers x 8 samples, whereas the original server spread them over 8 GPUs
# (DATA_PARALLEL_SIZE=8), so each request takes several times longer. With the
# script's 685s effective timeout, long generations would be cut off and
# retried. Only the client HTTP timeout is raised; sampling is unchanged.

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
# 65536 is the serve_qwen35.sh default context length.
start_vllm_server "${model_repo}" 65536 "logs/slurm/vllm-rollout-${SIZE}-${SLURM_JOB_ID:-local}.log"
wait_for_vllm_server

MODEL="qwen3.5-${SIZE}" BENCHMARKS="${BENCH}" BASE_URL="${VLLM_BASE_URL}" \
  bash scripts/run_rollout.sh

echo "done $(date -Is)"
