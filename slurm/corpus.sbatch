#!/bin/bash
#SBATCH --job-name=aggmath-corpus
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# CPU stage of the qwen3emb retriever. For each of the 15 rollout files it
# rebuilds the RAG passage corpus D_x and tokenizes passages and queries with
# the Qwen3-Embedding tokenizer, writing
#   outputs/SingleShot_RAG/embeddings/<model>/<bench>_thinking_n8.corpus.npz
# slurm/embed.sbatch (GPU) then only embeds these. Existing files are skipped.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
which python
python --version

python -m baselines.single_shot_rag.embed_passages --stage corpus \
  --inputs outputs/math_rollout/qwen3.5-2b outputs/math_rollout/qwen3.5-4b outputs/math_rollout/qwen3.5-9b

echo "done $(date -Is)"
