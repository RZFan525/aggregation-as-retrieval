#!/bin/bash
#SBATCH --job-name=aggmath-eval
#SBATCH --account=pi_hzamani_umass_edu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/pi_hzamani_umass_edu/runze/AggMath
#SBATCH --output=logs/slurm/%x-%j.out

# Pass@1 / Pass@8 for the 15 rollout files. CPU only; run after all rollouts are done.
# Same inputs and settings as scripts/run_eval.sh (its INPUTS block, K="1 8",
# OUTPUT_DIR=outputs/eval, plus WRITE_PER_PROBLEM=1 for the retrieval silver
# labels), with two differences:
# - aggmath.evaluation.run_eval runs through slurm/grader_timeout.py, which caps
#   each math_verify parse/verify at 60s (agreed 2026-10-25). The current
#   grader has no limit and hung on qwen3.5-4b aime26_013, rollout 4.
# - Each file gets its own process, because about 1,350 rollouts have no
#   \boxed{} answer and are graded on their full text, which can take up to 60s
#   each.
# Per-file results go to outputs/eval/<model>_<bench>_thinking_n8.eval.json(l).
# They are merged into outputs/eval/rollout_eval_summary.json in the same
# format as run_eval --summary-out.

set -euo pipefail
source /work/pi_hzamani_umass_edu/runze/AggMath/slurm/env.sh
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none}"
which python
python --version

mkdir -p outputs/eval
pids=()
for size in 2b 4b 9b; do
  for bench in aime24 aime25 aime26 hmmt24 hmmt25; do
    python slurm/grader_timeout.py aggmath.evaluation.run_eval \
      --inputs "outputs/math_rollout/qwen3.5-${size}/${bench}_thinking_n8.jsonl" \
      --k 1 8 \
      --output-dir outputs/eval \
      --write-per-problem \
      > "logs/slurm/eval-${SLURM_JOB_ID:-local}-qwen3.5-${size}-${bench}.log" 2>&1 &
    pids+=($!)
  done
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" != 0 ]]; then
  echo "At least one run_eval process failed; see logs/slurm/eval-${SLURM_JOB_ID:-local}-*.log" >&2
  exit 1
fi

python - << 'EOF'
import json
from pathlib import Path

files = []
for size in ["2b", "4b", "9b"]:
    for bench in ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]:
        path = Path("outputs/eval/qwen3.5-{}_{}_thinking_n8.eval.json".format(size, bench)).resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["_summary_path"] = str(path)
        files.append(summary)
        print("qwen3.5-{} {}: {}".format(size, bench, summary["pass_at_k"]))
Path("outputs/eval/rollout_eval_summary.json").write_text(
    json.dumps({"k": [1, 8], "files": files}, ensure_ascii=False, indent=2), encoding="utf-8"
)
EOF

echo "done $(date -Is)"
