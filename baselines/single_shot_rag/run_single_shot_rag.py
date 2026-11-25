"""Run single-shot RAG aggregation over a grouped rollout JSONL file.

Strategy (2) of the CS646 proposal:
1. Split the first k rollouts of a problem into a per-problem passage corpus D_x.
2. Retrieve the top-m passages once, using the problem statement as the query.
3. Have the aggregator answer from those passages.

Everything except the `{solutions}` text matches Prompted Aggregation
(ICA Full-CoT), because it reuses baselines.prompted_aggregation directly:
- prompt template and prompt-token budget;
- aggregator model and generation config (thinking, temperature, top_p, no
  top_k, no seed);
- client, answer extraction and output schema.
So baselines.prompted_aggregation.evaluate and
analysis/analyze_prompted_tokens.py work on these outputs unchanged.
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

from aggmath.rollout.qwen_api import (
    OpenAICompatClient,
    default_generation_config,
    normalize_model_name,
)
from aggmath.utils.answer_extraction import extract_boxed_answer
from baselines.prompted_aggregation.run_prompted_aggregation import (
    _budget_prompt,
    _build_token_usage,
    _dynamic_timeout,
    _extract_gold_answer,
    _parse_bool_flag,
    _read_completed_problem_ids,
    _read_jsonl,
    _read_rollout_metadata,
    _write_jsonl_line,
    build_output_path,
    load_token_counter,
)
from baselines.single_shot_rag.passages import (
    bm25_rank,
    build_passages,
    content_digest,
    dense_rank,
    format_retrieved_passages,
    passage_key,
)

RETRIEVERS = ("bm25", "qwen3emb")


def load_passage_tokenizer(model: str) -> Any:
    """The tokenizer AggAgent uses to split solutions into steps."""
    from vendor.aggagent_core.agent import AggAgent

    return AggAgent._load_tokenizer(AggAgent._default_tokenizer_name(model))


def load_embeddings(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read an embed_passages.py .npz file into per-problem passage/query vectors."""
    import numpy as np

    data = np.load(path, allow_pickle=False)
    by_problem: Dict[str, Dict[str, Any]] = {}
    for row, problem_id in enumerate(data["passage_problem_ids"]):
        entry = by_problem.setdefault(str(problem_id), {"rows": [], "keys": [], "digests": []})
        entry["rows"].append(row)
        entry["keys"].append((int(data["passage_solution_ids"][row]), int(data["passage_steps"][row])))
        entry["digests"].append(str(data["passage_digests"][row]))
    passage_vectors = data["passage_embeddings"]
    for entry in by_problem.values():
        entry["vectors"] = passage_vectors[entry.pop("rows")]
    for problem_id, query_vector in zip(data["query_problem_ids"], data["query_embeddings"]):
        by_problem.setdefault(str(problem_id), {"keys": [], "digests": []})["query"] = query_vector
    return by_problem


def retrieve(
    retriever: str,
    question: str,
    passages: Sequence[Dict[str, Any]],
    top_m: int,
    embedding_entry: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, float]]:
    if retriever == "bm25":
        return bm25_rank(question, passages, top_m)
    if retriever == "qwen3emb":
        if embedding_entry is None or "query" not in embedding_entry:
            raise ValueError("No embeddings for this problem; run embed_passages.py first.")
        keys = [passage_key(passage) for passage in passages]
        digests = [content_digest(str(passage["content"])) for passage in passages]
        if keys != embedding_entry["keys"] or digests != embedding_entry["digests"]:
            raise ValueError("Embedded passages do not match the rebuilt passage corpus.")
        return dense_rank(embedding_entry["query"], embedding_entry["vectors"], top_m)
    raise ValueError("Unknown retriever '{}'. Expected one of {}.".format(retriever, RETRIEVERS))


def _aggregate_record(
    record: Dict[str, Any],
    client: OpenAICompatClient,
    model: str,
    generation_config: Any,
    retriever: str,
    top_m: int,
    k: int,
    step_token_target: int,
    passage_tokenizer: Any,
    embedding_entry: Optional[Dict[str, Any]],
    max_prompt_tokens: int,
    token_counter: Any,
    max_retries: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list):
        raise ValueError("Record does not contain a rollouts list.")
    question = str(record.get("question") or "")
    passages = build_passages(rollouts, k, passage_tokenizer, step_token_target)
    ranked = retrieve(retriever, question, passages, top_m, embedding_entry)

    budgeted_prompt = _budget_prompt(
        problem=question,
        solutions=format_retrieved_passages(passages, ranked),
        max_prompt_tokens=max_prompt_tokens,
        token_counter=token_counter,
    )
    parsed = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": budgeted_prompt.prompt}],
        config=generation_config,
        n=1,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    response = parsed.choices[0].text
    retrieved = [
        {
            "rank": rank,
            "solution_id": int(passages[idx]["solution_id"]),
            "step": int(passages[idx]["step"]),
            "score": score,
            "tokens": int(passages[idx].get("tokens", 0) or 0),
        }
        for rank, (idx, score) in enumerate(ranked)
    ]
    return {
        "benchmark": record.get("benchmark"),
        "problem_id": record.get("problem_id"),
        "question": record.get("question"),
        "gold_answer": _extract_gold_answer(record),
        "retriever": retriever,
        "top_m": top_m,
        "k": k,
        "n_passages": len(passages),
        "retrieved": retrieved,
        "evidence_tokens": sum(item["tokens"] for item in retrieved),
        "aggregation_prompt": budgeted_prompt.prompt,
        "aggregated_response": response,
        "aggregated_answer": extract_boxed_answer(response),
        "prompt_truncated": budgeted_prompt.was_truncated,
        "token_usage": _build_token_usage(usage=parsed.usage or {}, budgeted_prompt=budgeted_prompt),
    }


def run_single_shot_rag(
    input_jsonl: Path,
    output_dir: Path,
    retriever: str,
    top_m: int = 16,
    embeddings_path: Optional[Path] = None,
    model: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    client: Optional[OpenAICompatClient] = None,
    k: int = 8,
    step_token_target: int = 1024,
    max_model_len: int = 262144,
    max_tokens: int = 24576,
    prompt_token_buffer: int = 1024,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_workers: int = 2,
    max_retries: int = 5,
    request_timeout: int = 300,
    overwrite: bool = False,
    token_counter: Optional[Any] = None,
    passage_tokenizer: Optional[Any] = None,
) -> Path:
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if retriever not in RETRIEVERS:
        raise ValueError("Unknown retriever '{}'. Expected one of {}.".format(retriever, RETRIEVERS))
    if k < 1 or top_m < 1:
        raise ValueError("k and top_m must be >= 1.")

    records = _read_jsonl(input_jsonl)
    if not records:
        raise ValueError("Input JSONL is empty: {}".format(input_jsonl))

    benchmark = str(records[0].get("benchmark") or "unknown")
    run_metadata = _read_rollout_metadata(input_jsonl)
    requested_model = model or str(run_metadata.get("model") or "qwen3.5-9b")
    api_model = normalize_model_name(requested_model)

    max_prompt_tokens = max_model_len - max_tokens - prompt_token_buffer
    if max_prompt_tokens <= 0:
        raise ValueError(
            "Invalid token budget: max_model_len ({}) must exceed max_tokens ({}) "
            "+ prompt_token_buffer ({}).".format(max_model_len, max_tokens, prompt_token_buffer)
        )

    embeddings: Dict[str, Dict[str, Any]] = {}
    if retriever == "qwen3emb":
        if embeddings_path is None:
            raise ValueError("--embeddings is required for the qwen3emb retriever.")
        embeddings = load_embeddings(Path(embeddings_path))

    if client is None:
        client = OpenAICompatClient(base_url=base_url, api_key=api_key, timeout_seconds=request_timeout)
    if token_counter is None:
        token_counter = load_token_counter(api_model)
    if passage_tokenizer is None:
        passage_tokenizer = load_passage_tokenizer(api_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(output_dir=output_dir, model=api_model, benchmark=benchmark, k=k)
    if overwrite and output_path.exists():
        output_path.unlink()
    done = set() if overwrite else _read_completed_problem_ids(output_path)

    tasks = [record for record in records if str(record.get("problem_id")) not in done]
    lock = threading.Lock()
    progress = tqdm(total=len(tasks), desc=output_path.name) if tqdm is not None else None

    generation_config = default_generation_config(
        thinking=thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=None,
    )
    timeout_seconds = _dynamic_timeout(max_tokens=max_tokens, base_timeout=request_timeout)

    def _worker(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        problem_id = str(record.get("problem_id"))
        try:
            output_record = _aggregate_record(
                record=record,
                client=client,
                model=api_model,
                generation_config=generation_config,
                retriever=retriever,
                top_m=top_m,
                k=k,
                step_token_target=step_token_target,
                passage_tokenizer=passage_tokenizer,
                embedding_entry=embeddings.get(problem_id),
                max_prompt_tokens=max_prompt_tokens,
                token_counter=token_counter,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
            )
            _write_jsonl_line(output_path, output_record, lock)
            return None
        except Exception as exc:  # noqa: BLE001
            return (problem_id, repr(exc))

    errors: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_worker, record) for record in tasks]
        for future in as_completed(futures):
            err = future.result()
            if err is not None:
                errors.append(err)
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    if errors:
        error_path = output_path.with_suffix(".errors.log")
        with error_path.open("a", encoding="utf-8") as fh:
            for problem_id, message in errors:
                fh.write("{}\t{}\n".format(problem_id, message))
        raise RuntimeError("{} aggregation requests failed. See {}".format(len(errors), error_path))

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-shot RAG aggregation over a grouped rollout JSONL file.")
    parser.add_argument("--input-jsonl", required=True, help="Grouped rollout JSONL file.")
    parser.add_argument("--retriever", required=True, choices=RETRIEVERS)
    parser.add_argument("--top-m", type=int, default=16, help="Number of passages retrieved per problem.")
    parser.add_argument("--embeddings", default=None, help="embed_passages.py .npz file (qwen3emb only).")
    parser.add_argument("--k", type=int, default=8, help="Number of rollouts in the passage corpus.")
    parser.add_argument("--step-token-target", type=int, default=1024, help="Passage size, as AggAgent's SOLUTION_STEP_TOKENS.")
    parser.add_argument(
        "--model",
        default=None,
        help="Aggregator model name or alias. Defaults to rollout metadata model or qwen3.5-9b.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", default="outputs/SingleShot_RAG")
    parser.add_argument("--max-model-len", type=int, default=262144)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--prompt-token-buffer", type=int, default=1024)
    parser.add_argument("--thinking", type=_parse_bool_flag, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = run_single_shot_rag(
        input_jsonl=Path(args.input_jsonl),
        output_dir=Path(args.output_dir),
        retriever=args.retriever,
        top_m=args.top_m,
        embeddings_path=Path(args.embeddings) if args.embeddings else None,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        k=args.k,
        step_token_target=args.step_token_target,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        prompt_token_buffer=args.prompt_token_buffer,
        thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        overwrite=args.overwrite,
    )
    print("Wrote aggregation output: {}".format(output_path), file=sys.stderr)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
