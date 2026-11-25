"""Accuracy of every method by model and benchmark, in percent.

Reads the evaluation files written by the repo evaluators (Pass@k, ICA,
agentic) and by the new baselines (majority voting, single-shot RAG). Every
accuracy is correct / 30 per benchmark and correct / 150 overall. A file whose
denominator is not 30 raises an error instead of being reported. Writes
outputs/analysis/accuracy.json and prints Markdown tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["2b", "4b", "9b"]
BENCHMARKS = ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]
N_PROBLEMS = 30
RAG_TOP_MS = [4, 8, 16, 32, 64, 128]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _count(path: Path) -> Optional[int]:
    summary = _load(path)
    if summary is None:
        return None
    if summary.get("n_total") != N_PROBLEMS:
        raise ValueError("{} has n_total={}, expected {}".format(path, summary.get("n_total"), N_PROBLEMS))
    return int(summary["n_correct"])


def method_paths(root: Path) -> Dict[str, str]:
    """Method name -> eval.json path pattern with {m} (model) and {b} (benchmark)."""
    methods = {
        "MV@8": "outputs/majority_voting/qwen3.5-{m}/{b}_thinking_n8_mv_k8.eval.json",
        "ICA Full-CoT": "outputs/Prompted_Aggregation/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.eval.json",
        "Agentic": "outputs/Aggent_Aggregation/qwen3.5-{m}/{b}_agentic_n8.eval.json",
        "RAG-Qwen3Emb (m=16)": "outputs/SingleShot_RAG/qwen3emb_m16/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.eval.json",
    }
    for top_m in RAG_TOP_MS:
        methods["RAG-BM25 (m={})".format(top_m)] = (
            "outputs/SingleShot_RAG/bm25_m{}/qwen3.5-{{m}}/qwen3.5-{{m}}_{{b}}_n8.eval.json".format(top_m)
        )
    return methods


def collect(root: Path) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """results[model][method][benchmark or 'overall'] = accuracy in percent (None = not run)."""
    results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for model in MODELS:
        rows: Dict[str, Dict[str, Optional[float]]] = {}
        pass_at_1: Dict[str, Optional[float]] = {}
        pass_at_8: Dict[str, Optional[float]] = {}
        for bench in BENCHMARKS:
            summary = _load(root / "outputs/eval/qwen3.5-{}_{}_thinking_n8.eval.json".format(model, bench))
            if summary is None:
                pass_at_1[bench] = pass_at_8[bench] = None
                continue
            if summary["n_problems"] != N_PROBLEMS or summary["n_rollouts_per_problem"] != 8:
                raise ValueError("unexpected Pass@k denominators for {} {}".format(model, bench))
            pass_at_1[bench] = 100.0 * summary["pass_at_k"]["pass@1"]
            pass_at_8[bench] = 100.0 * summary["pass_at_k"]["pass@8"]
        for name, values in (("Pass@1", pass_at_1), ("Pass@8 (oracle)", pass_at_8)):
            done = [v for v in values.values() if v is not None]
            values["overall"] = sum(done) / len(done) if len(done) == len(BENCHMARKS) else None
            rows[name] = values
        for name, pattern in method_paths(root).items():
            counts = {bench: _count(root / pattern.format(m=model, b=bench)) for bench in BENCHMARKS}
            values = {bench: (100.0 * c / N_PROBLEMS if c is not None else None) for bench, c in counts.items()}
            done = [c for c in counts.values() if c is not None]
            values["overall"] = (100.0 * sum(done) / (N_PROBLEMS * len(BENCHMARKS))
                                 if len(done) == len(BENCHMARKS) else None)
            if any(v is not None for v in values.values()):
                rows[name] = values
        results[model] = rows
    return results


def to_markdown(results: Dict[str, Dict[str, Dict[str, Optional[float]]]]) -> str:
    lines: List[str] = []
    for model, rows in results.items():
        lines.append("### Qwen3.5-{} (%)".format(model.upper()))
        lines.append("")
        lines.append("| Method | " + " | ".join(b.upper() for b in BENCHMARKS) + " | Overall |")
        lines.append("|---|" + "---|" * (len(BENCHMARKS) + 1))
        for name, values in rows.items():
            cells = ["{:.1f}".format(values[b]) if values.get(b) is not None else "-" for b in BENCHMARKS]
            overall = "**{:.1f}**".format(values["overall"]) if values.get("overall") is not None else "-"
            lines.append("| {} | {} | {} |".format(name, " | ".join(cells), overall))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--json-out", default="outputs/analysis/accuracy.json")
    parser.add_argument("--md-out", default="outputs/analysis/accuracy.md")
    args = parser.parse_args()
    root = Path(args.root)
    results = collect(root)
    markdown = to_markdown(results)
    for out, text in ((args.json_out, json.dumps(results, indent=2)), (args.md_out, markdown)):
        path = root / out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
