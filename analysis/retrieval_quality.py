"""Evidence quality of single-shot RAG and agentic aggregation (CS646 proposal, Section 6).

Silver label: a retrieved passage is correct if its source rollout reaches the
gold answer. Per-rollout correctness comes from run_eval's
outputs/eval/*.eval.jsonl.
- evidence precision: the fraction of retrieved passages that are correct;
- evidence coverage: whether at least one retrieved passage is correct.

Everything is reported on the "mixed" problems, where 1-7 of the 8 rollouts are
correct; on all-correct or all-wrong problems the silver label is constant.
Base rate: the fraction of correct rollouts, i.e. the expected precision of
retrieving solutions at random. Predictiveness: aggregation accuracy split by
coverage, and the AUC of precision for a correct aggregated answer.

Evidence of agentic aggregation:
- passage level: steps returned by get_reasoning_step and search_solution;
- solution level: those steps' solutions plus the solutions it summarized
  with get_solution_summary.
get_answer lists all 8 answers, so it is not counted as selective evidence.

Aggregated answers are re-graded with the repo grader, and the totals are
checked against the evaluator's n_correct. Writes
outputs/analysis/retrieval_quality.json and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aggmath.evaluation.grader import grade_prediction  # noqa: E402

MODELS = ["2b", "4b", "9b"]
BENCHMARKS = ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]
RAG_TOP_MS = [4, 8, 16, 32, 64, 128]


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def silver_labels(model: str, bench: str) -> Dict[str, List[bool]]:
    path = PROJECT_ROOT / "outputs/eval/qwen3.5-{}_{}_thinking_n8.eval.jsonl".format(model, bench)
    labels: Dict[str, List[bool]] = {}
    for entry in _records(path):
        flags = sorted(entry["rollouts"], key=lambda r: r["rollout_idx"])
        labels[str(entry["problem_id"])] = [bool(r["correct"]) for r in flags]
    return labels


def _passages_in(value: Any) -> Iterable[Tuple[int, int]]:
    """Every {solution_id, step} dict inside a parsed tool result."""
    if isinstance(value, dict):
        if "solution_id" in value and "step" in value:
            yield int(value["solution_id"]), int(value["step"])
        for child in value.values():
            yield from _passages_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from _passages_in(child)


def agentic_evidence(record: Dict[str, Any]) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    passages: Set[Tuple[int, int]] = set()
    summarized: Set[int] = set()
    for message in record.get("aggregation_messages") or []:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        name = message.get("name")
        if name in ("get_reasoning_step", "search_solution"):
            passages.update(_passages_in(result))
        elif name == "get_solution_summary" and isinstance(result, dict) and "solution_id" in result:
            summarized.add(int(result["solution_id"]))
    return passages, summarized


def _auc(scores: List[float], labels: List[bool]) -> Optional[float]:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    mixed = [r for r in rows if 0 < sum(r["silver"]) < len(r["silver"])]
    with_ev = [r for r in mixed if r["precision"] is not None]
    covered = [r for r in mixed if r["coverage"]]
    uncovered = [r for r in mixed if not r["coverage"]]
    return {
        "n_problems": len(rows),
        "n_mixed": len(mixed),
        "mean_evidence_items": _mean([float(r["n_items"]) for r in mixed]),
        "base_rate": _mean([sum(r["silver"]) / len(r["silver"]) for r in mixed]),
        "precision": _mean([r["precision"] for r in with_ev]),
        "coverage": _mean([1.0 if r["coverage"] else 0.0 for r in mixed]),
        "acc_if_covered": _mean([1.0 if r["correct"] else 0.0 for r in covered]),
        "n_covered": len(covered),
        "acc_if_not_covered": _mean([1.0 if r["correct"] else 0.0 for r in uncovered]),
        "n_not_covered": len(uncovered),
        "auc_precision": _auc([r["precision"] for r in with_ev], [r["correct"] for r in with_ev]),
    }


def _row(silver: List[bool], solutions: List[int], correct: bool) -> Dict[str, Any]:
    flags = [silver[sid] for sid in solutions if 0 <= sid < len(silver)]
    return {
        "silver": silver,
        "n_items": len(flags),
        "precision": _mean([1.0 if f else 0.0 for f in flags]),
        "coverage": any(flags),
        "correct": correct,
    }


def analyze_method(model: str, path_pattern: str, kind: str) -> Optional[Dict[str, Any]]:
    paths = [PROJECT_ROOT / path_pattern.format(m=model, b=b) for b in BENCHMARKS]
    if not all(p.exists() for p in paths):
        return None
    passage_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str], bool] = {}
    for bench, path in zip(BENCHMARKS, paths):
        labels = silver_labels(model, bench)
        n_correct = 0
        for record in _records(path):
            gold, answer = record.get("gold_answer"), record.get("aggregated_answer")
            correct = False
            if gold is not None and answer is not None:
                key = (str(gold), "\\boxed{{{}}}".format(answer))
                if key not in cache:
                    cache[key] = grade_prediction(*key)
                correct = cache[key]
            n_correct += correct
            silver = labels[str(record["problem_id"])]
            if kind == "rag":
                items = [(int(p["solution_id"]), int(p["step"])) for p in record.get("retrieved") or []]
                summarized: Set[int] = set()
            else:
                passages, summarized = agentic_evidence(record)
                items = sorted(passages)
            passage_rows.append(_row(silver, [sid for sid, _ in items], correct))
            solution_rows.append(_row(silver, sorted({sid for sid, _ in items} | summarized), correct))
        expected = json.loads(path.with_name(path.stem + ".eval.json").read_text())["n_correct"]
        if n_correct != expected:
            raise ValueError("{}: re-graded {} correct, evaluator says {}".format(path, n_correct, expected))
    return {"passage": summarize(passage_rows), "solution": summarize(solution_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", default="outputs/analysis/retrieval_quality.json")
    parser.add_argument("--md-out", default="outputs/analysis/retrieval_quality.md")
    args = parser.parse_args()
    methods = [("RAG-BM25 (m={})".format(m), "outputs/SingleShot_RAG/bm25_m{}/qwen3.5-{{m}}/qwen3.5-{{m}}_{{b}}_n8.jsonl".format(m), "rag")
               for m in RAG_TOP_MS]
    methods.append(("RAG-Qwen3Emb (m=16)", "outputs/SingleShot_RAG/qwen3emb_m16/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.jsonl", "rag"))
    methods.append(("Agentic", "outputs/Aggent_Aggregation/qwen3.5-{m}/{b}_agentic_n8.jsonl", "agentic"))

    results: Dict[str, Dict[str, Any]] = {}
    for model in MODELS:
        for name, pattern, kind in methods:
            out = analyze_method(model, pattern, kind)
            if out is not None:
                results.setdefault(model, {})[name] = out

    def pct(v: Optional[float]) -> str:
        return "-" if v is None else "{:.1f}".format(100 * v)

    lines = ["| Model | Method | Level | Mixed problems | Evidence items | Base rate | Precision | Coverage | Acc if covered (n) | Acc if not covered (n) | AUC(precision) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, per in results.items():
        for name, levels in per.items():
            for level in (("passage", "solution") if name == "Agentic" else ("passage",)):
                s = levels[level]
                lines.append("| {} | {} | {} | {} | {:.1f} | {} | {} | {} | {} ({}) | {} ({}) | {} |".format(
                    model.upper(), name, level, s["n_mixed"], s["mean_evidence_items"] or 0.0, pct(s["base_rate"]),
                    pct(s["precision"]), pct(s["coverage"]), pct(s["acc_if_covered"]), s["n_covered"],
                    pct(s["acc_if_not_covered"]), s["n_not_covered"],
                    "-" if s["auc_precision"] is None else "{:.2f}".format(s["auc_precision"])))
    markdown = "\n".join(lines)
    for out, text in ((args.json_out, json.dumps(results, indent=2)), (args.md_out, markdown)):
        path = PROJECT_ROOT / out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
