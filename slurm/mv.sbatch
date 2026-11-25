#!/bin/bash
#SBATCH --job-name=aggmath-mv
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Majority voting (k=8) over the 15 rollout files, written to
# outputs/majority_voting. CPU only; run after all rollouts are done.
# Only short \boxed{} answers are graded, so the full-text grading hang in
# slurm/eval.sbatch cannot happen here. It still runs through
# slurm/grader_timeout.py (60s math_verify limit) so it grades exactly like Pass@k.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
which python
python --version

python slurm/grader_timeout.py baselines.majority_voting.run_majority_voting \
  --inputs outputs/math_rollout/qwen3.5-2b outputs/math_rollout/qwen3.5-4b outputs/math_rollout/qwen3.5-9b \
  --k 8 \
  --output-dir outputs/majority_voting \
  --summary-out outputs/majority_voting/mv_summary.json

echo "done $(date -Is)"
