#!/bin/bash
#SBATCH --job-name=aggmath-analysis
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Accuracy tables, retrieval quality and aggregator token cost
# (analysis/collect_accuracy.py, retrieval_quality.py, aggregation_cost.py),
# written to outputs/analysis/. CPU only. Re-run it after new results arrive
# (e.g. the BM25 top-m sweep) to refresh the tables. Grading goes through
# slurm/grader_timeout.py (60s math_verify limit), as for Pass@k and MV.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
which python
python --version

python analysis/collect_accuracy.py
python slurm/grader_timeout.py analysis.retrieval_quality
python analysis/aggregation_cost.py

echo "done $(date -Is)"
