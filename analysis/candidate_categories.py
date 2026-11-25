"""Problems solved per candidate category (Table 2 of the final report).

A problem is "all8" when all 8 candidates are correct, "mixed" when some are,
and "none" when no candidate is. Totals are checked against each method's
.eval.json n_correct. Run from the repository root; set AGGMATH_ROOT to read
outputs from another checkout.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AGGMATH_ROOT", REPO))
sys.path.insert(0, str(REPO))
from aggmath.evaluation.grader import grade_prediction  # noqa: E402

BENCHES = ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]
METHODS = {
    **{f"RAG m={m}": f"outputs/SingleShot_RAG/bm25_m{m}/qwen3.5-{{m}}/qwen3.5-{{m}}_{{b}}_n8.jsonl" for m in (4, 16, 128)},
    "ICA": "outputs/Prompted_Aggregation/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.jsonl",
    "Agentic": "outputs/Aggent_Aggregation/qwen3.5-{m}/{b}_agentic_n8.jsonl",
}


def records(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def category(flags):
    n = sum(flags)
    return "all8" if n == len(flags) else ("none" if n == 0 else "mixed")


cache = {}
for model in ["2b", "4b", "9b"]:
    cats = {}
    for b in BENCHES:
        for e in records(ROOT / f"outputs/eval/qwen3.5-{model}_{b}_thinking_n8.eval.jsonl"):
            cats[(b, str(e["problem_id"]))] = category([bool(r["correct"]) for r in e["rollouts"]])
    sizes = {c: sum(1 for v in cats.values() if v == c) for c in ("all8", "mixed", "none")}
    print(f"\n{model.upper()}: problems per category {sizes}")
    for name, pattern in METHODS.items():
        counts = {"all8": 0, "mixed": 0, "none": 0}
        for b in BENCHES:
            path = ROOT / pattern.format(m=model, b=b)
            n_ok = 0
            for r in records(path):
                gold, ans = r.get("gold_answer"), r.get("aggregated_answer")
                ok = False
                if gold is not None and ans is not None:
                    key = (str(gold), "\\boxed{{{}}}".format(ans))
                    if key not in cache:
                        cache[key] = grade_prediction(*key)
                    ok = cache[key]
                n_ok += ok
                counts[cats[(b, str(r["problem_id"]))]] += ok
            expected = json.loads(path.with_name(path.stem + ".eval.json").read_text())["n_correct"]
            assert n_ok == expected, (path, n_ok, expected)
        total = sum(counts.values())
        print(f"  {name:<10} total {total:>3}/150  all8 {counts['all8']:>3}/{sizes['all8']}  "
              f"mixed {counts['mixed']:>3}/{sizes['mixed']}  none {counts['none']:>3}/{sizes['none']}")
