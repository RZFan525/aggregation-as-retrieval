#!/bin/bash
#SBATCH --job-name=aggmath-autoqueue
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/autoqueue/%x-%j.out

# Unattended driver for the AggMath experiment queue (see slurm/autoqueue.py).
# GPU policy (user, 2026-10-26): at most 4 AggMath jobs on the lab's superpod-a100;
# the public gpu and gpu-preempt partitions (A100-80G) have no cap. Jobs are
# preemption-requeued and the scripts resume. Tasks are submitted in this order:
#   rollout -> single-shot RAG (bm25) -> smoke test -> embeddings
#   -> RAG (qwen3emb) -> ICA Full-CoT -> agentic
#   -> BM25 top-m sweep (m = 32, 64, 128, 8, 4; added 2026-10-27)
# Once all rollouts are done it also submits the CPU jobs: majority voting,
# Pass@k eval and the RAG passage corpus (needed by the embeddings).
# Progress: logs/autoqueue/status.txt and this job's log.
# Stop:     touch logs/autoqueue/STOP   (or scancel this job)

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
which python

python slurm/autoqueue.py --superpod-max 4 --superpod-partition superpod-a100 \
  --public-partitions gpu-highvram,gpu-preempt-highvram --rag-top-m 16 --rag-sweep 32,64,128,8,4 --poll-seconds 300 \
  --max-runtime-hours 46
