# Running AggMath on Unity

These job scripts only add cluster plumbing: they start vLLM inside the job,
pick a port and set paths. Every experiment parameter comes from the repo's own
`scripts/*.sh` with their defaults, except the deviations listed at the end of
this file. See `documents/experiment_plan.md` for the full plan.

## Setup (done on 2026-10-24)

- **Conda env `aggmath`** (Python 3.11) at
  `/work/pi_hzamani_umass_edu/runze/miniconda3/envs/aggmath`.
  - vLLM 0.19.0 with torch 2.10.0+cu128. This is the April 2026 release closest
    to the nightly used originally.
  - `requirements.txt` was resolved with `uv pip install --exclude-newer 2026-05-06`.
  - rouge-score 0.1.2 comes from conda-forge, because its sdist cannot be built
    on the NFS-backed scratch.
- **Workspace** `/scratch3/workspace/runzefan_umass_edu-aggmath`, created with
  `ws_allocate aggmath 30`. Extend it with `ws_extend aggmath 30`.
  - `hf_cache/hub` holds the models (table below).
  - `hf_home/datasets` holds the five benchmarks.
  - `cache/` holds the vLLM, Triton and inductor compile caches.

| Model | Revision | Source |
|---|---|---|
| Qwen3.5-2B | `15852e8` | downloaded |
| Qwen3.5-4B | `851bf6e` | downloaded |
| Qwen3.5-9B | `c202236` | symlink to `/datasets/ai/qwen3/hub` |
| Qwen3-Embedding-8B | `1d8ad4c` | weights linked from `/datasets/ai`, tokenizer files downloaded |

The last commit to any of the Qwen3.5 repos was on 2026-03-02, so these are the
same weights as in the original April runs.

`slurm/env.sh` activates the env, points HF at the workspace cache and runs
offline (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`).

## Jobs

Each GPU job:
1. gets one A100-80G (`superpod-a100` first, then `gpu`);
2. starts a GPU watchdog: if utilization stays at 0% for 20 minutes (after a
   20-minute grace period), the job cancels itself. Unity flags GPUs idle for
   over an hour.
3. starts `scripts/serve_qwen35.sh` in the background on a free port, setting
   only `MODEL`, `PORT`, `DATA_PARALLEL_SIZE` (the job's GPU count) and
   `MAX_MODEL_LEN`;
4. waits for `/v1/models`;
5. runs the stage;
6. stops the server.

| Script | Runs | Resources |
|---|---|---|
| `rollout.sbatch` | `scripts/run_rollout.sh`, setting `MODEL`, `BENCHMARKS` and `BASE_URL` | GPU, context 65536 (serve default) |
| `rag.sbatch` | `scripts/run_single_shot_rag.sh` (`baselines/single_shot_rag`, new): same generation settings as ICA; `RETRIEVER=bm25\|qwen3emb`, `TOP_M` (default 16) | GPU, 262144 |
| `prompted.sbatch` | `scripts/run_prompted_aggregation_{4b,9b}.sh`; 2B uses the 9B wrapper with only `MODEL` and the paths swapped | GPU, 262144 |
| `agentic.sbatch` | `scripts/run_agentic_aggregation_{2b,4b,9b}.sh` | GPU, 262144 |
| `smoke.sbatch` | all four stages on 2B, aime24, first 2 problems, written to `outputs_smoke/` | GPU, 262144 |
| `corpus.sbatch` | `embed_passages --stage corpus`: build and tokenize the RAG passage corpora | CPU |
| `embed.sbatch` | `embed_passages --stage embed`: Qwen3-Embedding-8B vectors from the corpora (offline vLLM, exits on any error) | GPU |
| `eval.sbatch` | `aggmath.evaluation.run_eval` with the same inputs and settings as `scripts/run_eval.sh`, one process per file, 60s math_verify limit | CPU |
| `mv.sbatch` | majority voting (`baselines/majority_voting`, new), 60s math_verify limit | CPU |
| `analysis.sbatch` | `analysis/collect_accuracy.py`, `retrieval_quality.py` and `aggregation_cost.py`, writing to `outputs/analysis/` | CPU |
| `backup.sbatch` | snapshot of the whole AggMath directory into `/project/pi_hzamani_umass_edu/runze/backups/aggmath_<stamp>.tar.zst` with `.sha256` and a file list; `EXCLUDE_RUNNING=1` skips the sweep outputs still being written | CPU |

Logs:
- job log: `logs/slurm/<job-name>-<jobid>.out`
- vLLM log: `logs/slurm/vllm-*.log`
- the repo scripts' own logs: `logs/<stage>_<size>/`

## Submitting by hand

```bash
cd /work/pi_hzamani_umass_edu/runze/AggMath
bash slurm/submit.sh smoke        # end-to-end check first
bash slurm/submit.sh rollout      # 15 jobs: {2b,4b,9b} x 5 benchmarks
bash slurm/submit.sh eval         # after all rollouts are done
bash slurm/submit.sh prompted     # 15 jobs
bash slurm/submit.sh agentic      # 15 jobs

# Subset, plus extra sbatch arguments:
SIZES=9b BENCHES="aime24 hmmt25" bash slurm/submit.sh agentic --time=12:00:00

# The whole grid as one job array, at most 4 GPUs at a time, superpod only.
# This is how the rollouts were launched on 2026-10-25 (array 66041212):
SIZES="9b 4b 2b" BENCHES="hmmt24 hmmt25 aime24 aime25 aime26" ARRAY_MAX=4 \
  bash slurm/submit.sh rollout --partition=superpod-a100

# Print the sbatch commands without submitting:
DRY_RUN=1 bash slurm/submit.sh rollout
```

## Unattended queue (`autoqueue`)

`slurm/autoqueue.sbatch` is a 1-CPU job on `cpu,cpu-preempt`. It runs
`slurm/autoqueue.py`.
- It has a 2-day limit and resubmits itself after 46 h.
- It is requeued if preempted; its state is kept in `logs/autoqueue/state.json`.
- When every task is done, it submits `analysis.sbatch` and a full
  `backup.sbatch`, then exits.

**GPU policy** (set by the user on 2026-10-26):
- **superpod-a100** (the lab's partition): at most 4 AggMath jobs. This counts
  running jobs and pending jobs that may start there.
- **Public `gpu-highvram` and `gpu-preempt-highvram`** (A100-80G): no cap. Unity moved the A100-80G nodes out of `gpu`/`gpu-preempt` into these partitions around 2026-10-27; jobs that still asked for `gpu,gpu-preempt` with `a100-80g` sat in BadConstraints.
- **Partition choice**: superpod first. A job submitted while a superpod slot
  is free targets `superpod-a100` only; later jobs target `gpu,gpu-preempt`.
  When a superpod slot frees up, the highest-priority pending public job is
  moved to `superpod-a100`. Jobs that listed superpod together with the public
  partitions never started on superpod through preemption, which is why
  superpod jobs list superpod alone.
- **Preemption**: GPU jobs are submitted with `--requeue`, so a gpu-preempt
  preemption requeues the same job id and the scripts resume.

- **Order**: rollout → RAG (bm25, m=16) → smoke → embeddings → RAG (qwen3emb,
  m=16) → ICA Full-CoT → agentic → BM25 top-m sweep (`--rag-sweep 32,64,128,8,4`,
  added on 2026-10-27). Once all rollouts are done, it also submits
  the CPU jobs (MV, Pass@k eval, passage corpus). These do not count toward the
  4 GPUs.
- **Inputs**: a task is submitted only when its inputs are complete.
- **Completion**: a task is complete when its output JSONL has all 30
  problem_ids. For aggregation tasks, the `.eval.json` must also report
  `n_total == 30`, because the repo evaluators divide by the number of records,
  and a partial file would otherwise shrink the denominator.
- **Retries**: an incomplete task is resubmitted (the scripts resume) at most
  3 times. A stage is paused once 2 of its tasks are given up.
- **Gate**: ICA and agentic wait for the smoke test to pass.

```bash
sbatch slurm/autoqueue.sbatch                        # start
cat logs/autoqueue/status.txt                        # per-task status
tail logs/autoqueue/aggmath-autoqueue-<jobid>.out    # decisions
touch logs/autoqueue/STOP                            # stop at the next poll (or scancel)
python slurm/autoqueue.py --dry-run --once           # what it would do now
```

State (attempts, given-up tasks) is kept in `logs/autoqueue/state.json`.

Every stage resumes: re-submitting skips problem_ids that are already in the
output JSONL. Outputs go to the repo defaults: `outputs/math_rollout`,
`outputs/eval`, `outputs/Prompted_Aggregation` and `outputs/Aggent_Aggregation`.
The new methods write to `outputs/majority_voting` and `outputs/SingleShot_RAG`.

## Deviations from the script defaults (agreed with the user)

1. **`REQUEST_TIMEOUT=10800`** for rollout, ICA, RAG, agentic and smoke. Only
   the HTTP client timeout changes; sampling does not.
   - The script defaults give an effective 685s for rollout, 521s for ICA and
     300s per call for agentic.
   - The original server used 8 GPUs. Here one A100 serves all of a job's
     requests, so long generations hit those limits and were silently retried,
     which kept shorter answers. 9B ICA reached the 24k cap in only 4/150 calls.
   - The first ICA / RAG-bm25 / agentic-9B runs with the old timeout are kept in
     `outputs/_backup_timeout300_20261025` (logs in
     `logs/_backup_timeout300_20261025`). Everything was rerun from 2026-10-25.
2. **60s math_verify limit** for Pass@k and MV, via `slurm/grader_timeout.py`
   (the repo files are unchanged).
   - Since commit 9a04546 (2026-05-31), `grader.py` disables math_verify's
     timeouts. run_eval grades the full text of a rollout without `\boxed{}`, and
     one such text never finished.
   - The April 2026 grader used math_verify's default 5s limit. The user chose
     60s. A grading call that hits the limit counts as wrong.
   - The aggregation evaluators grade only short `\boxed{}` answers, so this
     never triggers for them.
