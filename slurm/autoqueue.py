#!/usr/bin/env python3
"""Keep the AggMath experiment queue busy on Unity GPUs.

This runs as slurm/autoqueue.sbatch, a 1-CPU job, so it keeps going while nobody
is logged in.

GPU policy (set by the user on 2026-10-26): use the lab's 4 superpod A100s first,
and queue the rest on the public partitions.
- The lab's superpod-a100 holds at most --superpod-max (4) AggMath jobs, counting
  both running jobs and pending jobs submitted there.
- A superpod job targets superpod-a100 alone. A job listing several partitions
  was never started there through preemption of gpu-preempt jobs.
- All other jobs go to the public partitions (--public-partitions, default
  gpu,gpu-preempt), which have no cap.
- When a superpod slot frees up, the highest-priority pending public job is
  moved to superpod-a100.

On every poll it:
1. lists this user's queued and running AggMath GPU jobs, with array tasks
   expanded and their partitions;
2. walks the task list in priority order and submits each task whose inputs are
   complete and whose own output is not, following the GPU policy above;
3. submits the CPU jobs (majority voting, Pass@k eval, RAG passage corpus) once
   all rollouts are complete. These do not count toward the GPU limit.

How tasks progress:
- A task is complete when its output JSONL has every problem_id. Aggregation
  stages also need their .eval.json. A job that timed out or crashed is simply
  resubmitted, and the repo scripts resume where they stopped.
- A task still incomplete after --max-attempts submissions is given up.
- A stage with --block-after given-up tasks is paused, so a systematic error
  cannot burn GPU hours unattended.
- ICA and agentic aggregation only start after the smoke test passes, i.e. both
  smoke problems got a final answer from each aggregator.

Stop it with `touch logs/autoqueue/STOP` (it exits at the next poll) or scancel.
Progress is printed to the job log and summarized in logs/autoqueue/status.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "logs" / "autoqueue"
SIZES = ["9b", "4b", "2b"]
BENCHES = ["hmmt24", "hmmt25", "aime24", "aime25", "aime26"]
N_PROBLEMS = 30
CPU_STAGES = {"eval", "mv", "corpus"}
SCRIPTS = {
    "rollout": "slurm/rollout.sbatch",
    "rag": "slurm/rag.sbatch",
    "smoke": "slurm/smoke.sbatch",
    "corpus": "slurm/corpus.sbatch",
    "embed": "slurm/embed.sbatch",
    "prompted": "slurm/prompted.sbatch",
    "agentic": "slurm/agentic.sbatch",
    "eval": "slurm/eval.sbatch",
    "mv": "slurm/mv.sbatch",
}
# Rollout arrays submitted by hand, mapping array index -> (size, bench).
# 66041212 used SIZES="9b 4b 2b" BENCHES="hmmt24 hmmt25 aime24 aime25 aime26".
ADOPTED_ROLLOUT_ARRAYS = {"66041212": [(size, bench) for size in SIZES for bench in BENCHES]}


@dataclass(frozen=True)
class Task:
    stage: str
    size: str = ""
    bench: str = ""
    retriever: str = ""
    top_m: int = 0

    @property
    def name(self) -> str:
        if self.stage == "rag":
            return "aggmath-rag-{}-m{}-{}-{}".format(self.retriever, self.top_m, self.size, self.bench)
        if self.size:
            return "aggmath-{}-{}-{}".format(self.stage, self.size, self.bench)
        return "aggmath-{}".format(self.stage)

    @property
    def stage_key(self) -> str:
        return "rag-{}".format(self.retriever) if self.stage == "rag" else self.stage


def build_tasks(top_m: int, sweep: Sequence[int] = ()) -> List[Task]:
    """All tasks in priority order (the order agreed in documents/experiment_plan.md).

    `sweep` adds BM25 single-shot RAG runs for extra top-m values (2026-10-27),
    after everything else, in the given order.
    """
    grid = [(size, bench) for size in SIZES for bench in BENCHES]
    tasks = [Task("rollout", s, b) for s, b in grid]
    tasks += [Task("rag", s, b, "bm25", top_m) for s, b in grid]
    tasks += [Task("mv"), Task("eval"), Task("smoke"), Task("corpus"), Task("embed")]
    tasks += [Task("rag", s, b, "qwen3emb", top_m) for s, b in grid]
    tasks += [Task("prompted", s, b) for s, b in grid]
    tasks += [Task("agentic", s, b) for s, b in grid]
    tasks += [Task("rag", s, b, "bm25", m) for m in sweep if m != top_m for s, b in grid]
    return tasks


# ---------------------------------------------------------------- output checks


def rollout_path(size: str, bench: str) -> Path:
    return ROOT / "outputs" / "math_rollout" / "qwen3.5-{}".format(size) / "{}_thinking_n8.jsonl".format(bench)


def output_path(task: Task) -> Optional[Path]:
    model_dir = "qwen3.5-{}".format(task.size)
    if task.stage == "rollout":
        return rollout_path(task.size, task.bench)
    if task.stage == "rag":
        return (ROOT / "outputs" / "SingleShot_RAG" / "{}_m{}".format(task.retriever, task.top_m) / model_dir
                / "{}_{}_n8.jsonl".format(model_dir, task.bench))
    if task.stage == "prompted":
        return ROOT / "outputs" / "Prompted_Aggregation" / model_dir / "{}_{}_n8.jsonl".format(model_dir, task.bench)
    if task.stage == "agentic":
        return ROOT / "outputs" / "Aggent_Aggregation" / model_dir / "{}_agentic_n8.jsonl".format(task.bench)
    return None


def embedding_path(size: str, bench: str, suffix: str = ".npz") -> Path:
    return (ROOT / "outputs" / "SingleShot_RAG" / "embeddings" / "qwen3.5-{}".format(size)
            / "{}_thinking_n8{}".format(bench, suffix))


def count_problem_ids(path: Path) -> int:
    if not path.exists():
        return 0
    ids = set()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                problem_id = json.loads(line).get("problem_id")
            except (json.JSONDecodeError, AttributeError):
                continue
            if problem_id is not None:
                ids.add(str(problem_id))
    return len(ids)


def read_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def smoke_passed() -> bool:
    """Both smoke aggregations ran end to end and produced at least one answer.

    A 2B model can legitimately miss one of the two problems, so this does not
    require both answers.
    """
    prompted = read_json(ROOT / "outputs_smoke" / "Prompted_Aggregation" / "qwen3.5-2b" / "qwen3.5-2b_aime24_n8.eval.json")
    agentic = read_json(ROOT / "outputs_smoke" / "Aggent_Aggregation" / "qwen3.5-2b" / "aime24_agentic_n8.eval.json")
    return (
        prompted.get("n_total") == 2
        and prompted.get("n_missing_answer", 2) < 2
        and agentic.get("n_total") == 2
        and agentic.get("n_empty_response", 2) < 2
    )


def is_complete(task: Task) -> bool:
    if task.stage == "smoke":
        return smoke_passed()
    if task.stage == "corpus":
        return all(embedding_path(s, b, ".corpus.npz").exists() for s in SIZES for b in BENCHES)
    if task.stage == "embed":
        return all(embedding_path(s, b).exists() for s in SIZES for b in BENCHES)
    if task.stage == "eval":
        return (ROOT / "outputs" / "eval" / "rollout_eval_summary.json").exists()
    if task.stage == "mv":
        return (ROOT / "outputs" / "majority_voting" / "mv_summary.json").exists()
    path = output_path(task)
    if path is None or count_problem_ids(path) < N_PROBLEMS:
        return False
    if task.stage == "rollout":
        return True
    # The repo evaluators divide by the number of records in the file, so a stale
    # .eval.json from a partial run could use a denominator below 30. Require 30.
    return read_json(path.with_name(path.stem + ".eval.json")).get("n_total") == N_PROBLEMS


# ------------------------------------------------------------------- the driver


class Driver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        sweep = [int(m) for m in args.rag_sweep.split(",") if m.strip()]
        self.tasks = build_tasks(args.rag_top_m, sweep)
        self.state_path = STATE_DIR / "state.json"
        self.state = read_json(self.state_path) or {}
        self.state.setdefault("done", [])
        self.state.setdefault("gave_up", [])
        self.state.setdefault("attempts", {})
        self.state.setdefault("jobs", {})
        self.done = set(self.state["done"])

    # -- helpers
    def log(self, message: str) -> None:
        print("[{}] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)

    def save(self) -> None:
        self.state["done"] = sorted(self.done)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def complete(self, task: Task) -> bool:
        if task.name in self.done:
            return True
        if is_complete(task):
            self.done.add(task.name)
            self.log("complete: {}".format(task.name))
            return True
        return False

    def stage_paused(self, task: Task) -> bool:
        given_up = [name for name in self.state["gave_up"] if self.task_stage_key(name) == task.stage_key]
        return len(given_up) >= self.args.block_after

    def task_stage_key(self, name: str) -> str:
        for task in self.tasks:
            if task.name == name:
                return task.stage_key
        return ""

    @staticmethod
    def prerequisites(task: Task) -> List[Task]:
        all_rollouts = [Task("rollout", s, b) for s in SIZES for b in BENCHES]
        if task.stage in ("rollout", "smoke"):
            return []
        if task.stage in ("eval", "mv", "corpus"):
            return all_rollouts
        if task.stage == "embed":
            return [Task("corpus")]
        prereqs = [Task("rollout", task.size, task.bench)]
        if task.stage == "rag" and task.retriever != "bm25":
            prereqs.append(Task("embed"))
        if task.stage in ("prompted", "agentic"):
            prereqs.append(Task("smoke"))
        return prereqs

    def done_now(self, task: Task, snapshot: Dict[str, List[str]]) -> bool:
        """Completion, checked at most once per poll and never while the task's job is queued."""
        if task.name not in self._done_cache:
            self._done_cache[task.name] = task.name not in snapshot and self.complete(task)
        return self._done_cache[task.name]

    def ready(self, task: Task, snapshot: Dict[str, List[str]]) -> bool:
        return all(self.done_now(prereq, snapshot) for prereq in self.prerequisites(task))

    def blocked(self, task: Task) -> bool:
        """True if a prerequisite (transitively) was given up, so this task can never run."""
        for prereq in self.prerequisites(task):
            if prereq.name in self.done:
                continue
            if prereq.name in self.state["gave_up"] or self.stage_paused(prereq) or self.blocked(prereq):
                return True
        return False

    def queue_snapshot(self) -> Optional[Dict[str, List[str]]]:
        """Map task name -> ['jobid:STATE:PARTITION', ...] for this user's queued/running jobs.

        A running job shows the partition it runs in. A pending job shows every
        partition it may start in.
        """
        try:
            result = subprocess.run(
                ["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i|%j|%T|%P"],
                capture_output=True, text=True, timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log("squeue failed: {}".format(exc))
            return None
        if result.returncode != 0:
            self.log("squeue failed: {}".format(result.stderr.strip()))
            return None
        snapshot: Dict[str, List[str]] = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 4:
                continue
            job_id, name, state, partition = parts
            base, _, index = job_id.partition("_")
            if base in ADOPTED_ROLLOUT_ARRAYS and index.isdigit():
                size, bench = ADOPTED_ROLLOUT_ARRAYS[base][int(index)]
                name = Task("rollout", size, bench).name
            snapshot.setdefault(name, []).append("{}:{}:{}".format(job_id, state, partition))
        return snapshot

    def submit(self, task: Task, partition: Optional[str] = None) -> Optional[str]:
        exports = ["ALL"]
        if task.size:
            exports += ["SIZE={}".format(task.size), "BENCH={}".format(task.bench)]
        if task.stage == "rag":
            exports += ["RETRIEVER={}".format(task.retriever), "TOP_M={}".format(task.top_m)]
        cmd = ["sbatch", "--parsable", "--job-name={}".format(task.name), "--export={}".format(",".join(exports))]
        if partition:
            # GPU job: requeue (same job id) if gpu-preempt preempts it.
            cmd += ["--partition={}".format(partition), "--requeue"]
        cmd.append(SCRIPTS[task.stage])
        if self.args.dry_run:
            self.log("DRY-RUN would submit: {}".format(" ".join(cmd)))
            return "dry-run"
        # Do not leak this job's SLURM_*/SBATCH_* settings into the child jobs.
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
        try:
            result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log("sbatch failed for {}: {}".format(task.name, exc))
            return None
        if result.returncode != 0:
            self.log("sbatch failed for {}: {}".format(task.name, result.stderr.strip()))
            return None
        job_id = result.stdout.strip().split(";")[0]
        self.log("submitted {} as job {}".format(task.name, job_id))
        return job_id

    def promote(self, job_id: str, partitions: str) -> bool:
        """Move a pending job to the given partition(s) (scontrol update)."""
        if self.args.dry_run:
            self.log("DRY-RUN would move pending job {} to {}".format(job_id, partitions))
            return True
        try:
            result = subprocess.run(["scontrol", "update", "JobId={}".format(job_id), "Partition={}".format(partitions)],
                                    capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.log("scontrol update failed for {}: {}".format(job_id, exc))
            return False
        if result.returncode != 0:
            self.log("scontrol update failed for {}: {}".format(job_id, result.stderr.strip()))
            return False
        self.log("moved pending job {} to {}".format(job_id, partitions))
        return True

    def write_status(self, snapshot: Dict[str, List[str]]) -> None:
        lines = ["updated {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), ""]
        for task in self.tasks:
            if task.name in self.done:
                status = "done"
            elif task.name in snapshot:
                status = " ".join(snapshot[task.name])
            elif task.name in self.state["gave_up"]:
                status = "GAVE UP after {} attempts".format(self.state["attempts"].get(task.name, 0))
            elif self.stage_paused(task):
                status = "stage paused"
            elif self.blocked(task):
                status = "blocked (a prerequisite was given up)"
            else:
                attempts = self.state["attempts"].get(task.name, 0)
                status = "waiting" + (" (attempts {})".format(attempts) if attempts else "")
            lines.append("{:48s} {}".format(task.name, status))
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- one poll
    def poll(self) -> bool:
        """Submit what fits. Return True once nothing is left to run or wait for."""
        snapshot = self.queue_snapshot()
        if snapshot is None:
            return False
        self._done_cache: Dict[str, bool] = {}
        # A GPU job is one queued on, or running in, a GPU partition. CPU jobs such
        # as eval, backup or analysis do not count.
        gpu_partitions = {self.args.superpod_partition} | set(self.args.public_partitions.split(","))
        gpu_jobs = [
            entry for name, jobs in snapshot.items() if name.startswith("aggmath-")
            for entry in jobs if gpu_partitions & set(entry.split(":", 2)[2].split(","))
        ]
        # The cap applies to the lab's superpod-a100 only: count jobs running or
        # pending there. The public partitions have no cap.
        superpod = self.args.superpod_partition
        superpod_busy = sum(1 for entry in gpu_jobs if superpod in entry.split(":", 2)[2].split(","))
        public_busy = len(gpu_jobs) - superpod_busy
        # Fill free superpod slots first: move the highest-priority pending public jobs there.
        for task in self.tasks:
            if task.stage in CPU_STAGES:
                continue
            for i, entry in enumerate(snapshot.get(task.name, [])):
                job_id, state, partition = entry.split(":", 2)
                if superpod_busy >= self.args.superpod_max:
                    break
                if state == "PENDING" and superpod not in partition.split(","):
                    if self.promote(job_id, superpod):
                        snapshot[task.name][i] = "{}:{}:{}".format(job_id, state, superpod)
                        superpod_busy += 1
                        public_busy -= 1
        open_tasks = 0
        for task in self.tasks:
            if task.name in snapshot:
                open_tasks += 1
                continue
            if self.done_now(task, snapshot):
                continue
            if task.name in self.state["gave_up"] or self.stage_paused(task) or self.blocked(task):
                continue
            open_tasks += 1
            if not self.ready(task, snapshot):
                continue
            attempts = self.state["attempts"].get(task.name, 0)
            if attempts >= self.args.max_attempts:
                self.state["gave_up"].append(task.name)
                self.log("GIVING UP on {} after {} attempts (check its logs)".format(task.name, attempts))
                if self.stage_paused(task):
                    self.log("PAUSING stage {}: {} tasks given up".format(task.stage_key, self.args.block_after))
                continue
            partition = None
            if task.stage not in CPU_STAGES:
                if superpod_busy < self.args.superpod_max:
                    partition = superpod
                    superpod_busy += 1
                elif public_busy < self.args.public_max:
                    partition = self.args.public_partitions
                    public_busy += 1
                else:
                    continue
            job_id = self.submit(task, partition)
            if job_id is None:
                if partition == superpod:
                    superpod_busy -= 1
                elif partition is not None:
                    public_busy -= 1
                continue
            if not self.args.dry_run:
                self.state["attempts"][task.name] = attempts + 1
                self.state["jobs"].setdefault(task.name, []).append(job_id)
            snapshot[task.name] = ["{}:SUBMITTED:{}".format(job_id, partition or "cpu")]
        self.write_status(snapshot)
        if not self.args.dry_run:
            self.save()
        self.log("poll: {} superpod-eligible + {} public GPU jobs queued/running, {} tasks open, {} done".format(
            superpod_busy, public_busy, open_tasks, len(self.done)))
        return open_tasks == 0

    def other_driver_running(self) -> bool:
        """True if another driver job is RUNNING; a pending one does not count."""
        own = os.environ.get("SLURM_JOB_ID", "")
        snapshot = self.queue_snapshot() or {}
        return any(
            entry.split(":")[1] == "RUNNING" and not entry.startswith(own + ":")
            for entry in snapshot.get("aggmath-autoqueue", []) if own
        )

    def submit_final_jobs(self) -> None:
        """Once everything is done: refresh the analysis tables and take a full backup."""
        if self.state.get("final_jobs_submitted"):
            return
        jobs = [["--partition=cpu,cpu-preempt", "slurm/analysis.sbatch"],
                ["--partition=cpu,cpu-preempt", "--export=ALL,EXCLUDE_RUNNING=0", "slurm/backup.sbatch"]]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
        for args in jobs:
            if self.args.dry_run:
                self.log("DRY-RUN would submit final job: sbatch {}".format(" ".join(args)))
                continue
            result = subprocess.run(["sbatch", "--parsable"] + args, cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=120)
            self.log("submitted final job {}: {} {}".format(args[-1], result.stdout.strip(), result.stderr.strip()))
        if not self.args.dry_run:
            self.state["final_jobs_submitted"] = True
            self.save()

    def resubmit_self(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
        result = subprocess.run(["sbatch", "--parsable", "slurm/autoqueue.sbatch"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=120)
        self.log("resubmitted driver: {} {}".format(result.stdout.strip(), result.stderr.strip()))

    def run(self) -> int:
        if not self.args.dry_run and self.other_driver_running():
            self.log("another aggmath-autoqueue job is running; exiting")
            return 0
        started = time.time()
        while True:
            if (STATE_DIR / "STOP").exists():
                self.log("STOP file found; exiting")
                return 0
            finished = self.poll()
            if finished:
                self.log("all tasks are done, given up or paused; submitting analysis + backup, then exiting")
                self.submit_final_jobs()
                return 0
            if self.args.once:
                return 0
            if time.time() - started > self.args.max_runtime_hours * 3600:
                self.resubmit_self()
                return 0
            time.sleep(self.args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keep the AggMath experiment queue busy on Unity GPUs.")
    parser.add_argument("--superpod-partition", default="superpod-a100", help="The lab partition with a GPU cap.")
    parser.add_argument("--superpod-max", type=int, default=4,
                        help="Max AggMath GPU jobs running on, or pending for, the superpod partition.")
    parser.add_argument("--public-partitions", default="gpu-highvram,gpu-preempt-highvram",
                        help="Comma-separated public partitions; jobs beyond the superpod cap go here.")
    parser.add_argument("--public-max", type=int, default=1000, help="Max GPU jobs on public partitions (effectively no cap).")
    parser.add_argument("--rag-top-m", type=int, default=16)
    parser.add_argument("--rag-sweep", default="", help="Extra BM25 top-m values to run, e.g. 32,64,128,8,4.")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--block-after", type=int, default=2, help="Pause a stage after this many given-up tasks.")
    parser.add_argument("--max-runtime-hours", type=float, default=160.0, help="Resubmit itself after this long.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted; change nothing.")
    parser.add_argument("--once", action="store_true", help="Run a single poll and exit.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return Driver(args).run()


if __name__ == "__main__":
    sys.exit(main())
