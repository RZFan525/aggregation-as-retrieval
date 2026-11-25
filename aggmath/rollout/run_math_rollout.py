"""Run text-only or thinking rollouts for math benchmarks with Qwen3.5."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

from .benchmark_loader import canonicalize_benchmark_name, load_benchmark, list_supported_benchmarks
from .prompts import build_math_messages
from .qwen_api import (
    OpenAICompatClient,
    default_generation_config,
    extract_boxed_answer,
    normalize_model_name,
)


def _safe_slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "_", lowered)
    return slug.strip("_") or "item"


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("Expected 'true' or 'false' for --thinking.")


def _thinking_label(thinking: bool) -> str:
    return "thinking" if thinking else "text"


def _model_dir_name(model: str) -> str:
    leaf = model.strip().split("/")[-1]
    return _safe_slug(leaf)


def _run_stem(benchmark: str, thinking_label: str, rollout_count: int) -> str:
    return "{}_{}_n{}".format(benchmark, thinking_label, rollout_count)


def _model_output_dir(output_root: Path, model: str) -> Path:
    return output_root / _model_dir_name(model)


def _jsonl_path(
    output_root: Path,
    benchmark: str,
    model: str,
    thinking_label: str,
    rollout_count: int,
) -> Path:
    return _model_output_dir(output_root, model) / "{}.jsonl".format(
        _run_stem(benchmark, thinking_label, rollout_count)
    )


def _metadata_path(
    output_root: Path,
    benchmark: str,
    model: str,
    thinking_label: str,
    rollout_count: int,
) -> Path:
    return _model_output_dir(output_root, model) / "{}.metadata.json".format(
        _run_stem(benchmark, thinking_label, rollout_count)
    )


def _error_log_path(
    output_root: Path,
    benchmark: str,
    model: str,
    thinking_label: str,
    rollout_count: int,
) -> Path:
    return _model_output_dir(output_root, model) / "{}.errors.log".format(
        _run_stem(benchmark, thinking_label, rollout_count)
    )


def _build_run_metadata(
    args: argparse.Namespace,
    benchmark: str,
    model: str,
    thinking: bool,
) -> Dict[str, Any]:
    return {
        "benchmark": benchmark,
        "model": model,
        "thinking": thinking,
        "mode": _thinking_label(thinking),
        "base_url": args.base_url,
        "rollout_count": args.rollout_count,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "base_seed": args.seed,
        "dataset_path": args.dataset_path,
        "dataset_root": args.dataset_root,
        "split": args.split,
        "longcot_split": args.longcot_split,
        "extra_instruction": args.extra_instruction,
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "request_timeout_base": args.request_timeout,
    }


def _read_existing_jsonl(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "rollouts" not in record and "rollout_idx" in record:
                raise ValueError(
                    "Existing file uses the legacy per-rollout JSONL schema: {}. "
                    "Re-run with --overwrite to regenerate it in the grouped schema.".format(path)
                )
            pid = record.get("problem_id")
            if pid is not None and isinstance(record.get("rollouts"), list):
                done.add(str(pid))
    return done


def _compute_seed(base_seed: Optional[int], problem_idx: int) -> Optional[int]:
    if base_seed is None:
        return None
    return base_seed + problem_idx


def _dynamic_timeout(max_tokens: int, base_timeout: int) -> int:
    return max(base_timeout, max_tokens // 50 + 30)


def _rollout_single_problem(
    client: OpenAICompatClient,
    benchmark: str,
    model: str,
    thinking: bool,
    problem: Any,
    problem_idx: int,
    args: argparse.Namespace,
    timeout_seconds: int,
) -> Dict[str, Any]:
    prompt_messages = build_math_messages(
        question=problem.question,
        benchmark=benchmark,
        instance=problem.to_instance_dict(),
        extra_instruction=args.extra_instruction,
    )
    request_seed = _compute_seed(args.seed, problem_idx)
    generation_config = default_generation_config(
        thinking=thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=request_seed,
    )

    t_start = time.perf_counter()
    parsed = client.chat_completion(
        model=model,
        messages=prompt_messages,
        config=generation_config,
        n=args.rollout_count,
        max_retries=args.max_retries,
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.perf_counter() - t_start
    usage = parsed.usage or {}
    cost_tokens = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }

    rollouts: List[Dict[str, Any]] = []
    for rollout_idx, choice in enumerate(parsed.choices):
        rollouts.append(
            {
                "rollout_idx": rollout_idx,
                "reasoning": choice.text,
                "answer": extract_boxed_answer(choice.text),
            }
        )

    return {
        "benchmark": benchmark,
        "problem_id": problem.question_id,
        "seed": request_seed,
        "question": problem.question,
        "instance": problem.to_instance_dict(),
        "prompt_messages": prompt_messages,
        "rollout_count_requested": args.rollout_count,
        "rollout_count_returned": len(rollouts),
        "rollouts": rollouts,
        "time": elapsed,
        "usage": usage,
        "cost_tokens": cost_tokens,
        "raw_response": parsed.raw_response,
    }


def _write_jsonl_line(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    payload = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")


def _write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def _build_task_list(
    problems: List[Any],
    done: Set[str],
) -> List[Tuple[Any, int]]:
    tasks: List[Tuple[Any, int]] = []
    for problem_idx, problem in enumerate(problems):
        if str(problem.question_id) in done:
            continue
        tasks.append((problem, problem_idx))
    return tasks


def _run_benchmark(
    args: argparse.Namespace,
    client: OpenAICompatClient,
    benchmark: str,
    model: str,
    thinking: bool,
    output_root: Path,
    per_call_timeout: int,
) -> None:
    thinking_label = _thinking_label(thinking)
    model_output_dir = _model_output_dir(output_root, model)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    if benchmark == "longcot_mini":
        split = args.longcot_split
    else:
        split = args.split

    dataset_path = args.dataset_path if len(args.benchmarks) == 1 else None
    problems = load_benchmark(
        benchmark=benchmark,
        dataset_path=dataset_path,
        dataset_root=args.dataset_root,
        split=split,
        max_problems=args.max_problems,
    )
    if args.start_index:
        problems = problems[args.start_index:]

    jsonl_path = _jsonl_path(output_root, benchmark, model, thinking_label, args.rollout_count)
    metadata_path = _metadata_path(output_root, benchmark, model, thinking_label, args.rollout_count)
    error_log = _error_log_path(output_root, benchmark, model, thinking_label, args.rollout_count)

    run_metadata = _build_run_metadata(args=args, benchmark=benchmark, model=model, thinking=thinking)
    _write_metadata(metadata_path, run_metadata)

    if args.overwrite and jsonl_path.exists():
        jsonl_path.unlink()
    done = set() if args.overwrite else _read_existing_jsonl(jsonl_path)

    tasks = _build_task_list(problems, done)
    if not tasks:
        return

    lock = threading.Lock()
    desc = "{}/{}".format(_model_dir_name(model), _run_stem(benchmark, thinking_label, args.rollout_count))
    progress = tqdm(total=len(tasks), desc=desc) if tqdm is not None else None

    def _worker(task: Tuple[Any, int]) -> Optional[Tuple[str, str]]:
        problem, problem_idx = task
        try:
            result = _rollout_single_problem(
                client=client,
                benchmark=benchmark,
                model=model,
                thinking=thinking,
                problem=problem,
                problem_idx=problem_idx,
                args=args,
                timeout_seconds=per_call_timeout,
            )
            _write_jsonl_line(jsonl_path, result, lock)
            return None
        except Exception as exc:  # noqa: BLE001
            return (str(problem.question_id), repr(exc))

    errors: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            err = future.result()
            if err is not None:
                errors.append(err)
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()

    if errors:
        with error_log.open("a", encoding="utf-8") as fh:
            for pid, msg in errors:
                fh.write("{}\t{}\n".format(pid, msg))


def run_rollout(args: argparse.Namespace) -> None:
    client = OpenAICompatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.request_timeout,
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmarks = [canonicalize_benchmark_name(name) for name in args.benchmarks]
    args.benchmarks = benchmarks
    model = normalize_model_name(args.model)
    per_call_timeout = _dynamic_timeout(args.max_tokens, args.request_timeout)

    for benchmark in benchmarks:
        _run_benchmark(
            args=args,
            client=client,
            benchmark=benchmark,
            model=model,
            thinking=args.thinking,
            output_root=output_root,
            per_call_timeout=per_call_timeout,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 rollouts over benchmark datasets.")
    parser.add_argument(
        "--model",
        required=True,
        help="A single model name or alias (e.g. 2b, 4b, 9b, or Qwen/Qwen3.5-9B).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="Benchmarks to run. Supported: {}".format(", ".join(list_supported_benchmarks())),
    )
    parser.add_argument(
        "--thinking",
        type=_parse_bool_flag,
        default=True,
        help="Whether to enable Qwen3.5 thinking mode: 'true' or 'false'.",
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--output-dir", default="outputs/math_rollout", help="Directory to store rollout JSONL files.")
    parser.add_argument("--dataset-path", default=None, help="Optional explicit .json/.jsonl dataset path. Use when running a single benchmark.")
    parser.add_argument("--dataset-root", default=None, help="Optional root directory for local benchmark files.")
    parser.add_argument("--split", default=None, help="Optional dataset split override for AIME/HMMT style loaders.")
    parser.add_argument(
        "--longcot-split",
        default=None,
        choices=["easy", "medium", "hard"],
        help="Optional split override for longcot_mini.",
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=1,
        help="Number of samples to generate per problem in a single request via the chat completions 'n' parameter.",
    )
    parser.add_argument("--max-problems", type=int, default=None, help="Optional cap on number of problems per benchmark.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional start index applied to every benchmark listed in --benchmarks.",
    )
    parser.add_argument("--max-tokens", type=int, default=65536, help="Maximum generation tokens per sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature. Defaults to 1.0.")
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p. Defaults to 0.95.")
    parser.add_argument("--top-k", type=int, default=20, help="top_k value passed via extra_body.")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Optional presence penalty override. Omitted from requests by default.",
    )
    parser.add_argument("--min-p", type=float, default=0.0, help="min_p value passed via extra_body.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="repetition_penalty value passed via extra_body. Defaults to 1.0.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed; per-rollout seed = base + rollout_idx (keeps K samples distinct).",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Baseline HTTP timeout (s). Effective per-call timeout = max(this, max_tokens/50 + 30).",
    )
    parser.add_argument("--extra-instruction", default=None, help="Optional extra instruction appended to the math prompt.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate JSONL before running. Default is resume: skip already-recorded (problem_id, rollout_idx).",
    )
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent in-flight requests against the vLLM server.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries on transient errors (5xx / 429 / timeout).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
