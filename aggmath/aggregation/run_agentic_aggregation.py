"""Run vendor AggAgent-style tool aggregation over grouped math rollouts."""

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

from aggmath.rollout.qwen_api import normalize_model_name
from aggmath.utils.answer_extraction import extract_boxed_answer
from vendor.aggagent_core.agent import AggAgent
from vendor.aggagent_core.tools import GetSolutionSummaryTool


@dataclass
class AgenticAggregationRunResult:
    output_path: Path


class _ConfiguredAggAgent(AggAgent):
    def __init__(self, *args: Any, max_tries: int = 5, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pipeline_max_tries = max_tries

    def call_server(self, messages, max_tries=5, only_finish=False):
        del max_tries
        return super().call_server(
            messages,
            max_tries=self._pipeline_max_tries,
            only_finish=only_finish,
        )


def build_litellm_kwargs(
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool,
    request_timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    return {
        "model": "hosted_vllm/{}".format(normalize_model_name(model)),
        "api_key": api_key,
        "api_base": base_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "parallel_tool_calls": False,
        "timeout": request_timeout,
        "num_retries": max_retries,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        },
    }


def build_agent(
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    summary_model: Optional[str] = None,
    summary_base_url: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    max_tokens: int = 10000,
    summary_max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.95,
    thinking: bool = True,
    request_timeout: int = 300,
    max_retries: int = 5,
    max_context_tokens: int = 100 * 1024,
    solution_step_tokens: int = 1024,
    tokenizer_name: Optional[str] = None,
    tokenizer: Any = None,
    search_tool: str = "rouge",
) -> AggAgent:
    api_model = normalize_model_name(model)
    summary_api_model = normalize_model_name(summary_model or model)
    aggregation_kwargs = build_litellm_kwargs(
        model=api_model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking=thinking,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )
    summary_kwargs = build_litellm_kwargs(
        model=summary_api_model,
        base_url=summary_base_url or base_url,
        api_key=summary_api_key or api_key,
        max_tokens=summary_max_tokens,
        temperature=0.2,
        top_p=0.95,
        thinking=False,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )

    agent = _ConfiguredAggAgent(
        model=api_model,
        api_base=base_url,
        max_context_tokens=max_context_tokens,
        llm_kwargs=aggregation_kwargs,
        solution_step_tokens=solution_step_tokens,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        summary_generation_kwargs={"max_tokens": summary_max_tokens},
        max_tries=max_retries,
        search_tool=search_tool,
    )
    agent.tool_map["get_solution_summary"] = GetSolutionSummaryTool(
        model=summary_api_model,
        api_base=summary_base_url or base_url,
        llm_kwargs=summary_kwargs,
        summary_generation_kwargs={"max_tokens": summary_max_tokens},
    )
    agent.tool_description = [
        agent.tool_map[name].get_tool_definitions() for name in agent.tool_map
    ]
    return agent


def build_output_path(output_dir: Path, model: str, benchmark: str, k: int) -> Path:
    return output_dir / _model_slug(model) / "{}_agentic_n{}.jsonl".format(
        _safe_slug(benchmark),
        k,
    )


def run_agentic_aggregation(
    input_jsonl: Path,
    output_dir: Path,
    model: Optional[str] = None,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    summary_model: Optional[str] = None,
    summary_base_url: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    agent: Optional[Any] = None,
    k: int = 8,
    max_context_tokens: int = 100 * 1024,
    solution_step_tokens: int = 1024,
    max_tokens: int = 10000,
    summary_max_tokens: int = 4096,
    thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_workers: int = 2,
    max_retries: int = 5,
    request_timeout: int = 300,
    overwrite: bool = False,
    tokenizer_name: Optional[str] = None,
    tokenizer: Any = None,
    search_tool: str = "rouge",
) -> AgenticAggregationRunResult:
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError("Input JSONL does not exist: {}".format(input_jsonl))
    if k < 1:
        raise ValueError("k must be >= 1.")

    records = _read_jsonl(input_jsonl)
    if not records:
        raise ValueError("Input JSONL is empty: {}".format(input_jsonl))

    benchmark = str(records[0].get("benchmark") or input_jsonl.stem)
    run_metadata = _read_rollout_metadata(input_jsonl)
    requested_model = model or str(run_metadata.get("model") or "qwen3.5-9b")
    api_model = normalize_model_name(requested_model)
    summary_api_model = normalize_model_name(summary_model or requested_model)

    output_path = build_output_path(output_dir, api_model, benchmark, k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_path.exists():
        output_path.unlink()
    done = set() if overwrite else _read_completed_problem_ids(output_path)
    tasks = [record for record in records if str(record.get("problem_id")) not in done]

    if agent is None:
        agent = build_agent(
            model=api_model,
            base_url=base_url,
            api_key=api_key,
            summary_model=summary_api_model,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            max_tokens=max_tokens,
            summary_max_tokens=summary_max_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
            request_timeout=request_timeout,
            max_retries=max_retries,
            max_context_tokens=max_context_tokens,
            solution_step_tokens=solution_step_tokens,
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer,
            search_tool=search_tool,
        )

    lock = threading.Lock()
    progress = tqdm(total=len(tasks), desc=output_path.name) if tqdm is not None else None

    def _worker(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        problem_id = str(record.get("problem_id"))
        try:
            output_record = _aggregate_record(
                record=record,
                agent=agent,
                aggregation_model=api_model,
                summary_model=summary_api_model,
                k=k,
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
            "{} agentic aggregation requests failed. See {}".format(
                len(errors), error_path
            )
        )

    return AgenticAggregationRunResult(output_path=output_path)


def _aggregate_record(
    record: Dict[str, Any],
    agent: Any,
    aggregation_model: str,
    summary_model: str,
    k: int,
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

    selected_rollouts = [rollout for rollout in rollouts[:k] if isinstance(rollout, dict)]
    if len(selected_rollouts) < k:
        raise ValueError("The first k rollouts must be JSON objects.")

    question = str(record.get("question") or "")
    agent_output = agent._run(question, selected_rollouts)
    result = agent_output.get("result") if isinstance(agent_output, dict) else None
    if not isinstance(result, dict):
        result = {}
    aggregated_response = _clean_optional_text(result.get("solution"))
    aggregated_answer = extract_boxed_answer(aggregated_response or "")

    return {
        "benchmark": record.get("benchmark"),
        "problem_id": record.get("problem_id"),
        "question": record.get("question"),
        "gold_answer": _extract_gold_answer(record),
        "rollouts": selected_rollouts,
        "aggregation_model": aggregation_model,
        "summary_model": summary_model,
        "solution_metadata": _extract_solution_metadata(agent_output),
        "aggregated_response": aggregated_response,
        "aggregated_answer": aggregated_answer,
        "aggregation_reason": _clean_optional_text(result.get("reason")),
        "aggregation_messages": agent_output.get("messages", []) if isinstance(agent_output, dict) else [],
        "aggregation_stats": agent_output.get("stats", {}) if isinstance(agent_output, dict) else {},
    }


def _extract_solution_metadata(agent_output: Any) -> Optional[str]:
    if not isinstance(agent_output, dict):
        return None
    messages = agent_output.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user_prompt = messages[1].get("content") if isinstance(messages[1], dict) else None
    if not isinstance(user_prompt, str):
        return None
    marker = "SOLUTION METADATA:\n"
    start = user_prompt.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = user_prompt.find("\n\nYou have", start)
    if end < 0:
        end = len(user_prompt)
    text = user_prompt[start:end].strip()
    return text or None


def _extract_gold_answer(record: Dict[str, Any]) -> Optional[str]:
    instance = record.get("instance") or {}
    answer = instance.get("answer") if isinstance(instance, dict) else None
    if answer is None:
        answer = record.get("answer")
    return _clean_optional_text(answer)


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
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
        description="Run agentic tool-based aggregation over grouped rollout JSONL."
    )
    parser.add_argument("--input-jsonl", required=True, help="Grouped rollout JSONL file.")
    parser.add_argument("--k", type=int, default=8, help="Number of rollouts to aggregate.")
    parser.add_argument(
        "--model",
        default=None,
        help="Aggregation model name or alias. Defaults to rollout metadata model or qwen3.5-9b.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--summary-model", default=None)
    parser.add_argument("--summary-base-url", default=None)
    parser.add_argument("--summary-api-key", default=None)
    parser.add_argument("--output-dir", default="outputs/Aggent_Aggregation")
    parser.add_argument("--max-context-tokens", type=int, default=100 * 1024)
    parser.add_argument("--solution-step-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--summary-max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", type=_parse_bool_flag, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument(
        "--search-tool",
        default="rouge",
        choices=["rouge", "bm25"],
        help="search_solution implementation: rouge keeps per-solution ROUGE-L search; bm25 uses bm25s to search all solution steps globally.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_agentic_aggregation(
        input_jsonl=Path(args.input_jsonl),
        output_dir=Path(args.output_dir),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        summary_model=args.summary_model,
        summary_base_url=args.summary_base_url,
        summary_api_key=args.summary_api_key,
        k=args.k,
        max_context_tokens=args.max_context_tokens,
        solution_step_tokens=args.solution_step_tokens,
        max_tokens=args.max_tokens,
        summary_max_tokens=args.summary_max_tokens,
        thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        overwrite=args.overwrite,
        tokenizer_name=args.tokenizer_name,
        search_tool=args.search_tool,
    )
    print("Wrote agentic aggregation output: {}".format(result.output_path), file=sys.stderr)
    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
