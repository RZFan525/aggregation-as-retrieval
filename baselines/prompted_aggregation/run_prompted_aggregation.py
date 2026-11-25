"""Run the Prompted Aggregation baseline over grouped rollout JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


AGGREGATION_PROMPT_TEMPLATE = (
    "Given the following problem:\n"
    "{problem}\n"
    "and these solution attempts:\n"
    "{solutions}\n"
    "It is possible that any, all, or none of these solutions are correct or complete. Carefully review the provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, and correct solution to the problem."
    "Please first provide a clear step-by-step reasoning process. After that, give the final answer, and put the final answer within \\boxed{{}}."
)


@dataclass
class BudgetedPrompt:
    prompt: str
    prompt_tokens: int
    original_prompt_tokens: int
    max_prompt_tokens: int
    was_truncated: bool
    kept_solution_chars: int
    original_solution_chars: int


class TransformersTokenCounter:
    """Tokenizer-backed counter for the final chat prompt sent to vLLM."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def count_prompt(self, prompt: str) -> int:
        messages = [{"role": "user", "content": prompt}]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                token_ids = apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return _count_tokenized_prompt(token_ids)
            except Exception:
                pass
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))


def _count_tokenized_prompt(tokenized: Any) -> int:
    input_ids = _extract_input_ids(tokenized)
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        shape_tuple = tuple(shape)
        if len(shape_tuple) >= 2:
            return int(shape_tuple[-1])
        if len(shape_tuple) == 1:
            return int(shape_tuple[0])

    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return len(input_ids[0])
        return len(input_ids)

    return len(input_ids)


def _extract_input_ids(tokenized: Any) -> Any:
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return tokenized["input_ids"]
    input_ids = getattr(tokenized, "input_ids", None)
    if input_ids is not None:
        return input_ids
    return tokenized


def load_token_counter(model: str) -> Any:
    normalized_model = normalize_model_name(model)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for Prompted Aggregation token counting. "
            "Install it or run inside the vLLM environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        normalized_model,
        trust_remote_code=True,
    )
    return TransformersTokenCounter(tokenizer)


def format_solution_attempts(rollouts: Sequence[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for idx, rollout in enumerate(rollouts, start=1):
        reasoning = rollout.get("reasoning")
        text = reasoning if isinstance(reasoning, str) else str(reasoning or "")
        chunks.append("Solution attempt {}:\n{}".format(idx, text.strip()))
    return "\n\n".join(chunks)


def build_aggregation_prompt(
    problem: str,
    rollouts: Sequence[Dict[str, Any]],
    k: int,
    max_prompt_tokens: int,
    token_counter: Any,
) -> BudgetedPrompt:
    selected_rollouts = list(rollouts[:k])
    solutions = format_solution_attempts(selected_rollouts)
    return _budget_prompt(
        problem=problem,
        solutions=solutions,
        max_prompt_tokens=max_prompt_tokens,
        token_counter=token_counter,
    )


def _budget_prompt(
    problem: str,
    solutions: str,
    max_prompt_tokens: int,
    token_counter: Any,
) -> BudgetedPrompt:
    if max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive.")

    original_prompt = _render_prompt(problem=problem, solutions=solutions)
    original_prompt_tokens = token_counter.count_prompt(original_prompt)
    if original_prompt_tokens <= max_prompt_tokens:
        return BudgetedPrompt(
            prompt=original_prompt,
            prompt_tokens=original_prompt_tokens,
            original_prompt_tokens=original_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
            was_truncated=False,
            kept_solution_chars=len(solutions),
            original_solution_chars=len(solutions),
        )

    empty_prompt = _render_prompt(problem=problem, solutions="")
    empty_prompt_tokens = token_counter.count_prompt(empty_prompt)
    if empty_prompt_tokens > max_prompt_tokens:
        raise ValueError(
            "Problem and aggregation instruction need {} tokens, which exceeds "
            "the prompt budget of {} tokens before adding any rollout text.".format(
                empty_prompt_tokens, max_prompt_tokens
            )
        )

    low = 0
    high = len(solutions)
    best_len = 0
    best_prompt = empty_prompt
    best_tokens = empty_prompt_tokens
    while low <= high:
        mid = (low + high) // 2
        candidate = _render_prompt(problem=problem, solutions=solutions[:mid])
        candidate_tokens = token_counter.count_prompt(candidate)
        if candidate_tokens <= max_prompt_tokens:
            best_len = mid
            best_prompt = candidate
            best_tokens = candidate_tokens
            low = mid + 1
        else:
            high = mid - 1

    while best_len > 0 and best_tokens > max_prompt_tokens:
        best_len -= 1
        best_prompt = _render_prompt(problem=problem, solutions=solutions[:best_len])
        best_tokens = token_counter.count_prompt(best_prompt)

    return BudgetedPrompt(
        prompt=best_prompt,
        prompt_tokens=best_tokens,
        original_prompt_tokens=original_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        was_truncated=True,
        kept_solution_chars=best_len,
        original_solution_chars=len(solutions),
    )


def _render_prompt(problem: str, solutions: str) -> str:
    return AGGREGATION_PROMPT_TEMPLATE.format(
        problem=problem.strip(),
        solutions=solutions.strip(),
    )


def build_output_path(output_dir: Path, model: str, benchmark: str, k: int) -> Path:
    return output_dir / "{}_{}_n{}.jsonl".format(
        _model_slug(model),
        _safe_slug(benchmark),
        k,
    )


def run_prompted_aggregation(
    input_jsonl: Path,
    output_dir: Path,
    model: Optional[str] = None,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    client: Optional[OpenAICompatClient] = None,
    k: int = 8,
    max_model_len: int = 65536,
    max_tokens: int = 16384,
    prompt_token_buffer: int = 1024,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_workers: int = 2,
    max_retries: int = 5,
    request_timeout: int = 300,
    overwrite: bool = False,
    token_counter: Optional[Any] = None,
) -> Path:
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError("Input JSONL does not exist: {}".format(input_jsonl))
    if k < 1:
        raise ValueError("k must be >= 1.")

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
            "+ prompt_token_buffer ({}).".format(
                max_model_len, max_tokens, prompt_token_buffer
            )
        )

    if client is None:
        client = OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=request_timeout,
        )
    if token_counter is None:
        token_counter = load_token_counter(api_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(
        output_dir=output_dir,
        model=api_model,
        benchmark=benchmark,
        k=k,
    )
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
                k=k,
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
        raise RuntimeError(
            "{} aggregation requests failed. See {}".format(len(errors), error_path)
        )

    return output_path


def _aggregate_record(
    record: Dict[str, Any],
    client: OpenAICompatClient,
    model: str,
    generation_config: Any,
    k: int,
    max_prompt_tokens: int,
    token_counter: Any,
    max_retries: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list):
        raise ValueError("Record does not contain a rollouts list.")
    if len(rollouts) < k:
        raise ValueError(
            "Record has only {} rollouts, fewer than requested k={}.".format(
                len(rollouts), k
            )
        )
    question = str(record.get("question") or "")
    budgeted_prompt = build_aggregation_prompt(
        problem=question,
        rollouts=rollouts,
        k=k,
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
    token_usage = _build_token_usage(
        usage=parsed.usage or {},
        budgeted_prompt=budgeted_prompt,
    )
    return {
        "benchmark": record.get("benchmark"),
        "problem_id": record.get("problem_id"),
        "question": record.get("question"),
        "gold_answer": _extract_gold_answer(record),
        "rollouts": rollouts,
        "aggregation_prompt": budgeted_prompt.prompt,
        "aggregated_response": response,
        "aggregated_answer": extract_boxed_answer(response),
        "prompt_truncated": budgeted_prompt.was_truncated,
        "token_usage": token_usage,
    }


def _build_token_usage(
    usage: Dict[str, Any],
    budgeted_prompt: BudgetedPrompt,
) -> Dict[str, int]:
    prompt_tokens = _int_or_default(usage.get("prompt_tokens"), budgeted_prompt.prompt_tokens)
    completion_tokens = _int_or_default(usage.get("completion_tokens"), 0)
    total_tokens = _int_or_default(
        usage.get("total_tokens"),
        prompt_tokens + completion_tokens,
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_prompt_tokens": budgeted_prompt.prompt_tokens,
        "original_estimated_prompt_tokens": budgeted_prompt.original_prompt_tokens,
        "max_prompt_tokens": budgeted_prompt.max_prompt_tokens,
    }


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _extract_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    instance = record.get("instance") or {}
    answer = instance.get("answer") if isinstance(instance, dict) else None
    if answer is None:
        answer = record.get("answer")
    if answer is None:
        return None
    text = str(answer).strip()
    return text or None


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


def _read_rollout_metadata(input_jsonl: Path) -> Dict[str, Any]:
    metadata_path = input_jsonl.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_completed_problem_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for record in _read_jsonl(path):
        problem_id = record.get("problem_id")
        if problem_id is not None:
            done.add(str(problem_id))
    return done


def _write_jsonl_line(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    payload = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")


def _dynamic_timeout(max_tokens: int, base_timeout: int) -> int:
    return max(base_timeout, max_tokens // 50 + 30)


def _safe_slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    return slug.strip("_") or "item"


def _model_slug(model: str) -> str:
    return _safe_slug(model.strip().split("/")[-1])


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected 'true' or 'false'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Prompted Aggregation over a grouped rollout JSONL file."
    )
    parser.add_argument("--input-jsonl", required=True, help="Grouped rollout JSONL file.")
    parser.add_argument("--k", type=int, default=8, help="Number of rollouts to aggregate.")
    parser.add_argument(
        "--model",
        default=None,
        help="Aggregator model name or alias. Defaults to rollout metadata model or qwen3.5-9b.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", default="outputs/Prompted_Aggregation")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=16384)
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
    output_path = run_prompted_aggregation(
        input_jsonl=Path(args.input_jsonl),
        output_dir=Path(args.output_dir),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        k=args.k,
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
