"""Aggregator token cost per method (CS646 proposal, Section 6).

Cost = input + output tokens of all aggregator calls, excluding the shared
cost of generating the 8 candidates (reported separately for reference).

- ICA Full-CoT and single-shot RAG: prompt_tokens / completion_tokens as
  reported by vLLM (token_usage in each record).
- Agentic, main loop: the per-iteration input/output tokens reported by vLLM
  (aggregation_stats.token_usage_each_step).
- Agentic, get_solution_summary sub-calls: the repo does not log their usage,
  so they are estimated.
  - Input: the chat-templated SUMMARY_PROMPT_TEMPLATE for that solution,
    counted with the model tokenizer (thinking off, as in the summary calls).
  - Output: the tokens of the summary returned to the agent. This is a lower
    bound, because the model also writes a "Reasoning:" part that the tool
    strips.
- Majority voting makes no model calls.

Writes outputs/analysis/cost.json and prints a Markdown table of mean tokens per
problem (thousands).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS = ["2b", "4b", "9b"]
BENCHMARKS = ["aime24", "aime25", "aime26", "hmmt24", "hmmt25"]
RAG_TOP_MS = [4, 8, 16, 32, 64, 128]


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _usage_cost(paths: List[Path]) -> Dict[str, float]:
    n = inp = out = 0
    for path in paths:
        for record in _records(path):
            usage = record.get("token_usage") or {}
            n += 1
            inp += int(usage.get("prompt_tokens", 0) or 0)
            out += int(usage.get("completion_tokens", 0) or 0)
    return {"n": n, "input": inp, "output": out}


def _summary_prompt_tokens(tokenizer: Any, question: str, answer: Any, reasoning: str) -> int:
    from vendor.aggagent_core.tools import SUMMARY_PROMPT_TEMPLATE

    prompt = SUMMARY_PROMPT_TEMPLATE.format(question=question, answer=answer, reasoning=reasoning)
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def _agentic_cost(paths: List[Path], tokenizer: Any) -> Dict[str, float]:
    n = main_in = main_out = sum_in = sum_out = n_summaries = 0
    for path in paths:
        for record in _records(path):
            n += 1
            for step in (record.get("aggregation_stats") or {}).get("token_usage_each_step", []):
                main_in += int(step.get("input_tokens") or 0)
                main_out += int(step.get("output_tokens") or 0)
            rollouts = record.get("rollouts") or []
            for message in record.get("aggregation_messages") or []:
                if message.get("role") != "tool" or message.get("name") != "get_solution_summary":
                    continue
                try:
                    result = json.loads(message.get("content") or "")
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(result, dict) or "solution_id" not in result:
                    continue
                sid = int(result["solution_id"])
                if not 0 <= sid < len(rollouts):
                    continue
                rollout = rollouts[sid]
                n_summaries += 1
                sum_in += _summary_prompt_tokens(tokenizer, str(record.get("question") or ""),
                                                 rollout.get("answer"), str(rollout.get("reasoning") or "").strip())
                sum_out += len(tokenizer.encode(str(result.get("summary") or ""), add_special_tokens=False))
    return {"n": n, "main_input": main_in, "main_output": main_out, "summary_calls": n_summaries,
            "summary_input_est": sum_in, "summary_output_est_lower_bound": sum_out,
            "input": main_in + sum_in, "output": main_out + sum_out}


def _generation_cost(paths: List[Path]) -> Dict[str, float]:
    n = out = 0
    for path in paths:
        for record in _records(path):
            n += 1
            out += int((record.get("cost_tokens") or {}).get("completion_tokens", 0) or 0)
    return {"n": n, "input": 0, "output": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json-out", default="outputs/analysis/cost.json")
    parser.add_argument("--md-out", default="outputs/analysis/cost.md")
    args = parser.parse_args()
    from vendor.aggagent_core.agent import AggAgent

    root = PROJECT_ROOT
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in MODELS:
        tokenizer = AggAgent._load_tokenizer(AggAgent._default_tokenizer_name("qwen3.5-{}".format(model)))
        per: Dict[str, Dict[str, float]] = {}
        per["Candidate generation (8 samples, reference)"] = _generation_cost(
            [root / "outputs/math_rollout/qwen3.5-{}/{}_thinking_n8.jsonl".format(model, b) for b in BENCHMARKS])
        per["MV@8"] = {"n": 150, "input": 0, "output": 0}
        for top_m in RAG_TOP_MS:
            paths = [root / "outputs/SingleShot_RAG/bm25_m{}/qwen3.5-{}/qwen3.5-{}_{}_n8.jsonl".format(top_m, model, model, b)
                     for b in BENCHMARKS]
            if all(p.exists() for p in paths):
                per["RAG-BM25 (m={})".format(top_m)] = _usage_cost(paths)
        paths = [root / "outputs/SingleShot_RAG/qwen3emb_m16/qwen3.5-{}/qwen3.5-{}_{}_n8.jsonl".format(model, model, b)
                 for b in BENCHMARKS]
        if all(p.exists() for p in paths):
            per["RAG-Qwen3Emb (m=16)"] = _usage_cost(paths)
        per["ICA Full-CoT"] = _usage_cost(
            [root / "outputs/Prompted_Aggregation/qwen3.5-{}/qwen3.5-{}_{}_n8.jsonl".format(model, model, b) for b in BENCHMARKS])
        per["Agentic"] = _agentic_cost(
            [root / "outputs/Aggent_Aggregation/qwen3.5-{}/{}_agentic_n8.jsonl".format(model, b) for b in BENCHMARKS], tokenizer)
        results[model] = per
        print("done", model, flush=True)

    lines = ["| Method | Model | Input/problem (k) | Output/problem (k) | Total/problem (k) | Notes |", "|---|---|---|---|---|---|"]
    for model, per in results.items():
        for name, c in per.items():
            n = max(int(c["n"]), 1)
            note = ""
            if name == "Agentic":
                note = "main loop {:.1f}k in / {:.1f}k out; {:.1f} summary calls/problem ≈ {:.1f}k in / ≥{:.1f}k out (estimated)".format(
                    c["main_input"] / n / 1e3, c["main_output"] / n / 1e3, c["summary_calls"] / n,
                    c["summary_input_est"] / n / 1e3, c["summary_output_est_lower_bound"] / n / 1e3)
            lines.append("| {} | {} | {:.1f} | {:.1f} | {:.1f} | {} |".format(
                name, model.upper(), c["input"] / n / 1e3, c["output"] / n / 1e3, (c["input"] + c["output"]) / n / 1e3, note))
    markdown = "\n".join(lines)
    for out, text in ((args.json_out, json.dumps(results, indent=2)), (args.md_out, markdown)):
        path = root / out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
