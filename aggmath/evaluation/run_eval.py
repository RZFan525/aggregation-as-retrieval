"""Evaluate rollout JSONL files with math_verify, computing pass@k.

Supports multiple input files and multiple k values in one invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

from .grader import grade_prediction
from .pass_at_k import pass_at_k
from ..utils.answer_extraction import extract_boxed_answer


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


def _extract_pred_text(rollout: Dict[str, Any]) -> str:
    answer = rollout.get("answer")
    if answer is not None:
        text = str(answer).strip()
        if text:
            return "\\boxed{{{}}}".format(text)

    reasoning = rollout.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        boxed = extract_boxed_answer(reasoning)
        if boxed:
            return "\\boxed{{{}}}".format(boxed)
        return reasoning
    return ""


def _extract_gold(record: Dict[str, Any]) -> Optional[str]:
    instance = record.get("instance") or {}
    answer = instance.get("answer")
    if answer is None:
        answer = record.get("answer")
    if answer is None:
        return None
    text = str(answer).strip()
    return text or None


def _grade_record(
    record: Dict[str, Any], cache: Dict[Tuple[str, str], bool]
) -> Dict[str, Any]:
    gold = _extract_gold(record)
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list):
        rollouts = []

    correctness: List[bool] = []
    per_rollout: List[Dict[str, Any]] = []
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            correctness.append(False)
            per_rollout.append({"rollout_idx": None, "correct": False, "answer": None})
            continue
        pred_text = _extract_pred_text(rollout)
        if gold is None or not pred_text:
            ok = False
        else:
            key = (gold, pred_text)
            if key in cache:
                ok = cache[key]
            else:
                ok = grade_prediction(gold, pred_text)
                cache[key] = ok
        correctness.append(ok)
        per_rollout.append(
            {
                "rollout_idx": rollout.get("rollout_idx"),
                "answer": rollout.get("answer"),
                "correct": ok,
            }
        )

    return {
        "problem_id": record.get("problem_id"),
        "gold": gold,
        "num_rollouts": len(correctness),
        "num_correct": sum(1 for c in correctness if c),
        "rollouts": per_rollout,
    }


def _summarize_file(
    path: Path,
    graded: List[Dict[str, Any]],
    k_values: List[int],
) -> Dict[str, Any]:
    n_problems = len(graded)
    if n_problems == 0:
        return {
            "file": str(path),
            "n_problems": 0,
            "n_rollouts_per_problem": None,
            "pass_at_k": {},
            "warnings": ["empty input"],
        }

    rollout_counts = sorted({entry["num_rollouts"] for entry in graded})
    warnings: List[str] = []
    if len(rollout_counts) > 1:
        warnings.append(
            "Mixed rollout counts within file: {}".format(rollout_counts)
        )
    n_min = rollout_counts[0]

    pass_at_k_summary: Dict[str, float] = {}
    for k in k_values:
        if k > n_min:
            warnings.append(
                "Skipping pass@{}: some problems have only {} rollouts.".format(k, n_min)
            )
            continue
        per_problem = [
            pass_at_k(entry["num_rollouts"], entry["num_correct"], k)
            for entry in graded
        ]
        pass_at_k_summary["pass@{}".format(k)] = sum(per_problem) / len(per_problem)

    return {
        "file": str(path),
        "n_problems": n_problems,
        "n_rollouts_per_problem": (
            n_min if len(rollout_counts) == 1 else rollout_counts
        ),
        "pass_at_k": pass_at_k_summary,
        "warnings": warnings,
    }


def _evaluate_file(
    path: Path,
    k_values: List[int],
    output_dir: Optional[Path],
    write_per_problem: bool,
) -> Dict[str, Any]:
    records = _read_jsonl(path)
    cache: Dict[Tuple[str, str], bool] = {}

    iterator: Any = records
    if tqdm is not None:
        iterator = tqdm(records, desc=path.name, unit="problem")

    graded: List[Dict[str, Any]] = []
    for record in iterator:
        graded.append(_grade_record(record, cache))

    summary = _summarize_file(path, graded, k_values)

    target_dir = output_dir if output_dir is not None else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _eval_output_stem(path, output_dir)
    summary_path = target_dir / (output_stem + ".eval.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    if write_per_problem:
        per_problem_path = target_dir / (output_stem + ".eval.jsonl")
        with per_problem_path.open("w", encoding="utf-8") as fh:
            for entry in graded:
                fh.write(json.dumps(entry, ensure_ascii=False))
                fh.write("\n")

    summary["_summary_path"] = str(summary_path)
    return summary


def _eval_output_stem(path: Path, output_dir: Optional[Path]) -> str:
    if output_dir is None:
        return path.stem

    parts = path.parts
    if "math_rollout" in parts:
        idx = parts.index("math_rollout")
        tail = parts[idx + 1 :]
        if len(tail) >= 2:
            return "{}_{}".format(tail[0], path.stem)
    return path.stem


def _expand_input_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if any(ch in raw for ch in "*?["):
            matches = sorted(Path().glob(raw)) or sorted(p.parent.glob(p.name))
            if not matches:
                raise FileNotFoundError("No files matched glob: {}".format(raw))
            paths.extend(matches)
        elif p.is_dir():
            matches = sorted(p.glob("*.jsonl"))
            if not matches:
                raise FileNotFoundError("No .jsonl files in directory: {}".format(p))
            paths.extend(matches)
        else:
            if not p.exists():
                raise FileNotFoundError("Input file not found: {}".format(p))
            paths.append(p)
    return paths


def _print_table(summaries: List[Dict[str, Any]], k_values: List[int]) -> None:
    headers = ["file", "n", "rollouts"] + ["pass@{}".format(k) for k in k_values]
    rows: List[List[str]] = []
    for summary in summaries:
        row = [
            _display_file_name(summary["file"]),
            str(summary["n_problems"]),
            str(summary["n_rollouts_per_problem"]),
        ]
        for k in k_values:
            key = "pass@{}".format(k)
            value = summary["pass_at_k"].get(key)
            row.append("-" if value is None else "{:.4f}".format(value))
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def _display_file_name(path_value: str) -> str:
    path = Path(path_value)
    parts = path.parts
    if "math_rollout" in parts:
        idx = parts.index("math_rollout")
        tail = parts[idx + 1 :]
        if tail:
            return str(Path(*tail))
    if path.parent != Path("."):
        return str(Path(path.parent.name) / path.name)
    return path.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade rollout JSONL with math_verify and report pass@k."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more rollout JSONL files (or directories containing .jsonl, "
            "or globs)."
        ),
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1],
        help="One or more k values for pass@k. Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write .eval.json (and optionally .eval.jsonl). "
        "Default: alongside each input file.",
    )
    parser.add_argument(
        "--write-per-problem",
        action="store_true",
        help="Also write a per-problem .eval.jsonl with rollout-level grades.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to dump a single combined summary JSON.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    k_values = sorted(set(int(k) for k in args.k))
    if any(k < 1 for k in k_values):
        parser.error("--k values must be >= 1.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    )
    paths = _expand_input_paths(args.inputs)

    summaries: List[Dict[str, Any]] = []
    for path in paths:
        print("{} start".format(path), flush=True)
        summary = _evaluate_file(
            path=path,
            k_values=k_values,
            output_dir=output_dir,
            write_per_problem=args.write_per_problem,
        )
        summaries.append(summary)

    for summary in summaries:
        for w in summary.get("warnings", []):
            print(
                "[warn] {}: {}".format(Path(summary["file"]).name, w),
                file=sys.stderr,
                flush=True,
            )

    print()
    _print_table(summaries, k_values)
    print()

    if args.summary_out:
        out = Path(args.summary_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump({"k": k_values, "files": summaries}, fh, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
