"""Extra statistics for the CS646 final report, computed from saved outputs only.

1. Per-problem correctness for agentic, ICA, RAG-BM25 (m=16, m=128); totals are
   checked against each file's .eval.json n_correct.
2. Exact McNemar tests between methods on the same 150 problems per model.
3. Agent behaviour: turns, tool calls per type, forced finishes.
4. Problems where no candidate is correct: how many have unfinished candidates.

Run from the repository root; set AGGMATH_ROOT to read outputs from another checkout.
"""
import json
import os
import sys
from collections import Counter
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("AGGMATH_ROOT", REPO))
sys.path.insert(0, str(REPO))
from aggmath.evaluation.grader import grade_prediction  # noqa: E402

BENCHES = ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]
METHODS = {
    "RAG16": "outputs/SingleShot_RAG/bm25_m16/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.jsonl",
    "RAG128": "outputs/SingleShot_RAG/bm25_m128/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.jsonl",
    "ICA": "outputs/Prompted_Aggregation/qwen3.5-{m}/qwen3.5-{m}_{b}_n8.jsonl",
    "Agent": "outputs/Aggent_Aggregation/qwen3.5-{m}/{b}_agentic_n8.jsonl",
}


def records(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


cache = {}
for model in ["2b", "4b", "9b"]:
    cand = {}
    for b in BENCHES:
        for e in records(ROOT / f"outputs/eval/qwen3.5-{model}_{b}_thinking_n8.eval.jsonl"):
            cand[(b, str(e["problem_id"]))] = e["rollouts"]
    correct = {}
    agent_stats = []
    for name, pattern in METHODS.items():
        flags = {}
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
                    ok = bool(cache[key])
                flags[(b, str(r["problem_id"]))] = ok
                n_ok += ok
                if name == "Agent":
                    agent_stats.append(r.get("aggregation_stats", {}))
            expected = json.loads(path.with_name(path.stem + ".eval.json").read_text())["n_correct"]
            assert n_ok == expected, (path, n_ok, expected)
        assert len(flags) == 150, (name, model, len(flags))
        correct[name] = flags
    print(f"\n=== {model.upper()} ===")
    print("accuracy:", {k: f"{sum(v.values())}/150" for k, v in correct.items()})
    for a, c in [("Agent", "ICA"), ("Agent", "RAG128"), ("Agent", "RAG16"), ("ICA", "RAG16"), ("RAG128", "ICA")]:
        wins = sum(correct[a][k] and not correct[c][k] for k in correct[a])
        losses = sum(correct[c][k] and not correct[a][k] for k in correct[a])
        print(f"  {a} vs {c}: wins {wins}, losses {losses}, McNemar exact p = {mcnemar_exact(wins, losses):.4f}")
    # agent behaviour
    n = len(agent_stats)
    tools = Counter()
    for s in agent_stats:
        tools.update(s.get("tool_calls", {}))
    iters = [s.get("iterations", 0) for s in agent_stats]
    forced = sum(bool(s.get("context_limit_reached")) for s in agent_stats)
    no_search = sum(1 for s in agent_stats
                    if not s.get("tool_calls", {}).get("search_solution") and not s.get("tool_calls", {}).get("get_reasoning_step"))
    print(f"  agent: mean turns {sum(iters)/n:.1f} (max {max(iters)}), forced finish {forced}/{n}, "
          f"no search/step reads {no_search}/{n}")
    print("  tool calls per problem:", {k: round(v / n, 2) for k, v in sorted(tools.items())})
    # unfinished candidates on none-correct problems
    none = [k for k, rs in cand.items() if not any(r["correct"] for r in rs)]
    def unfinished(rs):
        return sum(1 for r in rs if r.get("answer") in (None, ""))
    some_unf = sum(1 for k in none if unfinished(cand[k]) > 0)
    all_unf = sum(1 for k in none if unfinished(cand[k]) == 8)
    agent_none_wins = [k for k in none if correct["Agent"][k]]
    print(f"  none-correct problems {len(none)}: >=1 unfinished {some_unf}, all 8 unfinished {all_unf}")
    print(f"  agent solved {len(agent_none_wins)} of them; of those >=1 unfinished "
          f"{sum(1 for k in agent_none_wins if unfinished(cand[k]) > 0)}, all unfinished "
          f"{sum(1 for k in agent_none_wins if unfinished(cand[k]) == 8)}")
    total_unf = sum(unfinished(rs) for rs in cand.values())
    print(f"  candidates without a final answer: {total_unf}/1200")
