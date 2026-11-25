"""Evaluate agentic aggregation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aggmath.evaluation.grader import grade_prediction


def evaluate_agentic_aggregation_file(
    input_jsonl: Path,
    output_dir: Optional[Path] = None,
    write_per_problem: bool = False,
) -> Dict[str, Any]:
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    records = _read_jsonl(input_jsonl)
    cache: Dict[Tuple[str, str], bool] = {}

    graded = [_grade_aggregation_record(record, cache) for record in records]
    n_total = len(graded)
    n_correct = sum(1 for entry in graded if entry["correct"])
    n_missing = sum(1 for entry in graded if entry["missing_answer"])
    n_empty_response = sum(1 for entry in graded if entry["empty_response"])
    n_context_limit = sum(1 for entry in graded if entry["context_limit_reached"])
    n_tool_call_errors = sum(int(entry["tool_call_errors"] or 0) for entry in graded)

    summary = {
        "file": str(input_jsonl),
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_total - n_correct,
        "n_missing_answer": n_missing,
        "n_empty_response": n_empty_response,
        "n_context_limit_reached": n_context_limit,
        "n_tool_call_errors": n_tool_call_errors,
        "accuracy": (float(n_correct) / n_total) if n_total else 0.0,
    }

    target_dir = output_dir.expanduser().resolve() if output_dir else input_jsonl.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "{}.eval.json".format(input_jsonl.stem)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if write_per_problem:
        per_problem_path = target_dir / "{}.eval.jsonl".format(input_jsonl.stem)
        with per_problem_path.open("w", encoding="utf-8") as fh:
            for entry in graded:
                fh.write(json.dumps(entry, ensure_ascii=False))
                fh.write("\n")

    summary["_summary_path"] = str(summary_path)
    return summary


def _grade_aggregation_record(
    record: Dict[str, Any],
    cache: Dict[Tuple[str, str], bool],
) -> Dict[str, Any]:
    gold = _clean_optional_text(record.get("gold_answer"))
    answer = _clean_optional_text(record.get("aggregated_answer"))
    response = _clean_optional_text(record.get("aggregated_response"))
    stats = record.get("aggregation_stats") if isinstance(record.get("aggregation_stats"), dict) else {}
    context_limit_reached = bool(stats.get("context_limit_reached"))
    tool_call_errors = _int_or_default(stats.get("tool_call_errors"), 0)

    missing = answer is None
    pred_text = "\\boxed{{{}}}".format(answer) if answer is not None else None
    if gold is None or pred_text is None:
        correct = False
    else:
        key = (gold, pred_text)
        if key not in cache:
            cache[key] = grade_prediction(gold, pred_text)
        correct = cache[key]
    return {
        "problem_id": record.get("problem_id"),
        "gold_answer": gold,
        "aggregated_answer": answer,
        "correct": correct,
        "missing_answer": missing,
        "empty_response": response is None,
        "context_limit_reached": context_limit_reached,
        "tool_call_errors": tool_call_errors,
    }


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Malformed JSON in {}:{}: {}".format(path, line_idx, exc)
                ) from exc
    return records


def _print_summary(summary: Dict[str, Any]) -> None:
    print(
        "file={} n={} correct={} wrong={} missing={} empty_response={} "
        "context_limit={} tool_call_errors={} accuracy={:.4f}".format(
            Path(summary["file"]).name,
            summary["n_total"],
            summary["n_correct"],
            summary["n_wrong"],
            summary["n_missing_answer"],
            summary["n_empty_response"],
            summary["n_context_limit_reached"],
            summary["n_tool_call_errors"],
            summary["accuracy"],
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate agentic aggregation JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Agentic aggregation JSONL.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .eval.json outputs. Default: next to input.",
    )
    parser.add_argument(
        "--write-per-problem",
        action="store_true",
        help="Also write a per-problem .eval.jsonl.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = evaluate_agentic_aggregation_file(
        Path(args.input_jsonl),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        write_per_problem=args.write_per_problem,
    )
    _print_summary(summary)
    print("Wrote evaluation summary: {}".format(summary["_summary_path"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
