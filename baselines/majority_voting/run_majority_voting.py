"""Majority Voting over grouped rollout JSONL files.

Each rollout's extracted \\boxed{} answer counts as one vote, and rollouts
without one abstain. Votes are grouped by symbolic equivalence using the same
math_verify grader that scores rollouts
(aggmath.evaluation.grader.grade_prediction). The largest group wins; on a tie,
the group whose first vote has the lowest rollout index wins. The winning
answer is graded exactly like a rollout answer in aggmath.evaluation.run_eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

from aggmath.evaluation.grader import grade_prediction
from aggmath.evaluation.run_eval import _expand_input_paths, _extract_gold, _read_jsonl

EquivalenceFn = Callable[[str, str], bool]


def boxed(answer: str) -> str:
    return "\\boxed{{{}}}".format(answer)


def collect_votes(rollouts: Sequence[Any], k: int) -> List[Tuple[int, str]]:
    """Return (rollout_idx, answer) for the first k rollouts that have an answer."""
    votes: List[Tuple[int, str]] = []
    for position, rollout in enumerate(list(rollouts)[:k]):
        if not isinstance(rollout, dict):
            continue
        answer = rollout.get("answer")
        text = str(answer).strip() if answer is not None else ""
        if not text:
            continue
        rollout_idx = rollout.get("rollout_idx")
        votes.append((int(rollout_idx) if rollout_idx is not None else position, text))
    return votes


def cluster_votes(
    votes: Sequence[Tuple[int, str]],
    equivalent: EquivalenceFn,
) -> List[Dict[str, Any]]:
    """Group votes into equivalence classes, keyed by each class's first answer."""
    clusters: List[Dict[str, Any]] = []
    for rollout_idx, answer in votes:
        for cluster in clusters:
            if cluster["answer"] == answer or equivalent(cluster["answer"], answer):
                cluster["rollout_idxs"].append(rollout_idx)
                break
        else:
            clusters.append({"answer": answer, "rollout_idxs": [rollout_idx]})
    for cluster in clusters:
        cluster["count"] = len(cluster["rollout_idxs"])
    return clusters


def pick_majority(clusters: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not clusters:
        return None
    return max(clusters, key=lambda c: (c["count"], -min(c["rollout_idxs"])))


def make_equivalence(cache: Dict[Tuple[str, str], bool]) -> EquivalenceFn:
    def equivalent(reference: str, answer: str) -> bool:
        key = (reference, answer)
        if key not in cache:
            cache[key] = grade_prediction(reference, boxed(answer))
        return cache[key]

    return equivalent


def vote_record(
    record: Dict[str, Any],
    k: int,
    equivalent: EquivalenceFn,
    grade_cache: Dict[Tuple[str, str], bool],
) -> Dict[str, Any]:
    gold = _extract_gold(record)
    rollouts = record.get("rollouts")
    votes = collect_votes(rollouts if isinstance(rollouts, list) else [], k)
    clusters = cluster_votes(votes, equivalent)
    winner = pick_majority(clusters)
    answer = winner["answer"] if winner is not None else None

    correct = False
    if gold is not None and answer is not None:
        key = (gold, boxed(answer))
        if key not in grade_cache:
            grade_cache[key] = grade_prediction(gold, boxed(answer))
        correct = grade_cache[key]

    return {
        "benchmark": record.get("benchmark"),
        "problem_id": record.get("problem_id"),
        "gold_answer": gold,
        "k": k,
        "n_votes": len(votes),
        "clusters": sorted(clusters, key=lambda c: (-c["count"], min(c["rollout_idxs"]))),
        "majority_answer": answer,
        "majority_count": winner["count"] if winner is not None else 0,
        "correct": correct,
    }


def build_output_stem(input_jsonl: Path, k: int) -> str:
    return "{}_mv_k{}".format(input_jsonl.stem, k)


def run_file(input_jsonl: Path, output_dir: Path, k: int) -> Dict[str, Any]:
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    records = _read_jsonl(input_jsonl)
    equivalence_cache: Dict[Tuple[str, str], bool] = {}
    grade_cache: Dict[Tuple[str, str], bool] = {}
    equivalent = make_equivalence(equivalence_cache)

    iterator: Any = records
    if tqdm is not None:
        iterator = tqdm(records, desc=input_jsonl.name, unit="problem")
    results = [vote_record(record, k, equivalent, grade_cache) for record in iterator]

    target_dir = Path(output_dir).expanduser().resolve() / input_jsonl.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = build_output_stem(input_jsonl, k)
    output_path = target_dir / "{}.jsonl".format(stem)
    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False))
            fh.write("\n")

    n_total = len(results)
    n_correct = sum(1 for result in results if result["correct"])
    summary = {
        "file": str(input_jsonl),
        "output": str(output_path),
        "k": k,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_total - n_correct,
        "n_no_vote": sum(1 for result in results if result["majority_answer"] is None),
        "accuracy": (float(n_correct) / n_total) if n_total else 0.0,
    }
    summary_path = target_dir / "{}.eval.json".format(stem)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _print_table(summaries: Sequence[Dict[str, Any]]) -> None:
    headers = ["file", "n", "k", "mv_acc", "no_vote"]
    rows = [
        [
            str(Path(summary["file"]).parent.name + "/" + Path(summary["file"]).name),
            str(summary["n_total"]),
            str(summary["k"]),
            "{:.4f}".format(summary["accuracy"]),
            str(summary["n_no_vote"]),
        ]
        for summary in summaries
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Majority Voting over grouped rollout JSONL files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Rollout JSONL files, directories (all *.jsonl inside) or globs.",
    )
    parser.add_argument("--k", type=int, default=8, help="Vote over the first k rollouts.")
    parser.add_argument("--output-dir", default="outputs/majority_voting")
    parser.add_argument("--summary-out", default=None, help="Optional combined summary JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1.")

    inputs = [path for path in _expand_input_paths(args.inputs) if not path.name.endswith(".eval.jsonl")]
    summaries = [run_file(path, Path(args.output_dir), args.k) for path in inputs]
    _print_table(summaries)
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summaries, fh, ensure_ascii=False, indent=2)
        print("Wrote combined summary: {}".format(summary_path), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
