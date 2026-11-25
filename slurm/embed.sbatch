#!/bin/bash
#SBATCH --job-name=aggmath-embed
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# GPU stage of the qwen3emb retriever: embed the passage corpora that
# slurm/corpus.sbatch built, with Qwen3-Embedding-8B in offline vLLM (no
# server). All CPU work (passage splitting and tokenization) was done in the
# corpus job, so the GPU is busy from the start. Output, one file per rollout file:
#   outputs/SingleShot_RAG/embeddings/<model>/<bench>_thinking_n8.npz
# Files that already exist are skipped. The script exits on its first error
# instead of hanging.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
print_job_info
start_gpu_watchdog 15 10

python -m baselines.single_shot_rag.embed_passages --stage embed \
  --inputs outputs/math_rollout/qwen3.5-2b outputs/math_rollout/qwen3.5-4b outputs/math_rollout/qwen3.5-9b

echo "done $(date -Is)"
