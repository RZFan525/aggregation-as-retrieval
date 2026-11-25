#!/usr/bin/env bash
# Submit AggMath jobs on Unity.
#
#   bash slurm/submit.sh rollout|rag|prompted|agentic [extra sbatch args...]
#       One job per (model size, benchmark). Narrow the grid with
#       SIZES="2b 4b 9b" and BENCHES="aime24 aime25 aime26 hmmt24 hmmt25".
#       For rag, set RETRIEVER=bm25|qwen3emb and TOP_M=16 in the environment.
#   bash slurm/submit.sh eval|embed|smoke [extra sbatch args...]
#
# ARRAY_MAX=N submits the grid as one job array with at most N tasks running
# at once (one GPU each). Tasks start in SIZES x BENCHES order.
# DRY_RUN=1 prints the sbatch commands without submitting.
# Example: SIZES=9b BENCHES=aime24 DRY_RUN=1 bash slurm/submit.sh rollout --time=02:00:00

set -euo pipefail

AGGMATH_ROOT="${AGGMATH_ROOT:-/work/pi_hzamani_umass_edu/runze/AggMath}"
cd "${AGGMATH_ROOT}"
mkdir -p logs/slurm

stage="${1:?Usage: bash slurm/submit.sh <rollout|rag|prompted|agentic|eval|embed|smoke> [sbatch args...]}"
shift
SIZES="${SIZES:-2b 4b 9b}"
BENCHES="${BENCHES:-aime24 aime25 aime26 hmmt24 hmmt25}"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "$*"
  else
    "$@"
  fi
}

case "${stage}" in
  rollout | rag | prompted | agentic)
    if [[ -n "${ARRAY_MAX:-}" ]]; then
      tasks=()
      for size in ${SIZES}; do
        for bench in ${BENCHES}; do
          tasks+=("${size}:${bench}")
        done
      done
      run sbatch \
        --job-name="aggmath-${stage}" \
        --array="0-$((${#tasks[@]} - 1))%${ARRAY_MAX}" \
        --output="logs/slurm/%x-%A_%a.out" \
        --export="ALL,TASKS=${tasks[*]}" \
        "$@" \
        "slurm/${stage}.sbatch"
    else
      for size in ${SIZES}; do
        for bench in ${BENCHES}; do
          run sbatch \
            --job-name="aggmath-${stage}-${size}-${bench}" \
            --export="ALL,SIZE=${size},BENCH=${bench}" \
            "$@" \
            "slurm/${stage}.sbatch"
        done
      done
    fi
    ;;
  eval | embed | smoke)
    run sbatch "$@" "slurm/${stage}.sbatch"
    ;;
  *)
    echo "Unknown stage '${stage}'. Expected rollout, rag, prompted, agentic, eval, embed or smoke." >&2
    exit 1
    ;;
esac
