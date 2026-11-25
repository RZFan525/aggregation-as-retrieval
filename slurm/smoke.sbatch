#!/bin/bash
#SBATCH --job-name=aggmath-smoke
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=superpod-a100,gpu
#SBATCH --constraint=a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# End-to-end smoke test on Qwen3.5-2B, aime24, first 2 problems: rollout, eval,
# prompted aggregation and agentic aggregation, all through the repo's own entry
# points with the same parameters as the full runs. Everything is written under
# outputs_smoke/ and logs/smoke/, so the real outputs/ stay untouched (the
# rollout resume logic would otherwise skip these 2 problems later).

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
source slurm/vllm_server.sh
# Same client timeout as the full runs (see rollout/prompted/agentic.sbatch).
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-10800}"
print_job_info
start_gpu_watchdog

SMOKE=outputs_smoke
ROLLOUT_DIR="${SMOKE}/math_rollout/qwen3.5-2b"
mkdir -p "${SMOKE}" logs/smoke

# One server for all stages; 262144 is what the aggregation stages use.
start_vllm_server "Qwen/Qwen3.5-2B" 262144 "logs/slurm/vllm-smoke-${SLURM_JOB_ID:-local}.log"
wait_for_vllm_server

echo "== [1/4] rollout $(date -Is)"
# Same arguments as scripts/run_rollout.sh plus --max-problems 2, which that
# wrapper does not expose.
python -m aggmath.rollout.run_math_rollout \
  --model qwen3.5-2b \
  --benchmarks aime24 \
  --thinking true \
  --base-url "${VLLM_BASE_URL}" \
  --api-key EMPTY \
  --output-dir "${SMOKE}/math_rollout" \
  --rollout-count 8 \
  --max-tokens 32768 \
  --max-workers 16 \
  --max-retries 5 \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --max-problems 2

echo "== [2/4] eval $(date -Is)"
python -m aggmath.evaluation.run_eval \
  --inputs "${ROLLOUT_DIR}/aime24_thinking_n8.jsonl" \
  --k 1 8 \
  --output-dir "${SMOKE}/eval" \
  --write-per-problem

echo "== [3/4] prompted aggregation $(date -Is)"
MODEL="Qwen/Qwen3.5-2B" \
  INPUT_DIR="${ROLLOUT_DIR}" \
  OUTPUT_DIR="${SMOKE}/Prompted_Aggregation/qwen3.5-2b" \
  LOG_DIR="logs/smoke/prompted_aggregation_2b" \
  BASE_URL="${VLLM_BASE_URL}" \
  bash scripts/run_prompted_aggregation_9b.sh aime24

echo "== [4/4] agentic aggregation $(date -Is)"
INPUT_DIR="${ROLLOUT_DIR}" \
  OUTPUT_DIR="${SMOKE}/Aggent_Aggregation" \
  LOG_DIR="logs/smoke/agentic_aggregation_2b" \
  BASE_URL="${VLLM_BASE_URL}" \
  bash scripts/run_agentic_aggregation_2b.sh aime24

echo "== summary"
python - << 'EOF'
import glob
import json

with open("outputs_smoke/math_rollout/qwen3.5-2b/aime24_thinking_n8.jsonl") as fh:
    for line in fh:
        rec = json.loads(line)
        answers = [r.get("answer") for r in rec["rollouts"]]
        print("rollout", rec["problem_id"], "returned", rec["rollout_count_returned"],
              "completion_tokens", rec["cost_tokens"]["completion_tokens"], "answers", answers)
for path in sorted(glob.glob("outputs_smoke/**/*.eval.json", recursive=True)):
    with open(path) as fh:
        print(path, json.load(fh))
EOF

echo "done $(date -Is)"
