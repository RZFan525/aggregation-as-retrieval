"""Benchmark loading and normalization for math rollout experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..utils.answer_extraction import extract_boxed_answer


@dataclass
class MathProblem:
    """Normalized math problem record used by rollout code."""

    benchmark: str
    question_id: str
    question: str
    answer: Optional[str]
    raw: Dict[str, Any]

    def to_instance_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "raw": self.raw,
        }


@dataclass
class BenchmarkSpec:
    canonical_name: str
    hf_repo: Optional[str]
    hf_subset: Optional[str]
    default_split: str


_BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "aime24": BenchmarkSpec(
        canonical_name="aime24",
        hf_repo="math-ai/aime24",
        hf_subset=None,
        default_split="test",
    ),
    "aime25": BenchmarkSpec(
        canonical_name="aime25",
        hf_repo="MathArena/aime_2025",
        hf_subset=None,
        default_split="train",
    ),
    "aime26": BenchmarkSpec(
        canonical_name="aime26",
        hf_repo="MathArena/aime_2026",
        hf_subset=None,
        default_split="train",
    ),
    "hmmt24": BenchmarkSpec(
        canonical_name="hmmt24",
        hf_repo="MathArena/hmmt_feb_2024",
        hf_subset=None,
        default_split="train",
    ),
    "hmmt25": BenchmarkSpec(
        canonical_name="hmmt25",
        hf_repo="MathArena/hmmt_feb_2025",
        hf_subset=None,
        default_split="train",
    ),
    "hmmt25_nov": BenchmarkSpec(
        canonical_name="hmmt25_nov",
        hf_repo="MathArena/hmmt_nov_2025",
        hf_subset=None,
        default_split="train",
    ),
    "longcot_mini": BenchmarkSpec(
        canonical_name="longcot_mini",
        hf_repo="LongHorizonReasoning/longcot",
        hf_subset="all",
        default_split="easy",
    ),
}

_ALIASES = {
    "aime_2024": "aime24",
    "aime-2024": "aime24",
    "aime24": "aime24",
    "aime_2025": "aime25",
    "aime-2025": "aime25",
    "aime25": "aime25",
    "aime26": "aime26",
    "aime_2026": "aime26",
    "aime-2026": "aime26",
    "hmmt_2024": "hmmt24",
    "hmmt-2024": "hmmt24",
    "hmmt24": "hmmt24",
    "hmmt_2025": "hmmt25",
    "hmmt-2025": "hmmt25",
    "hmmt25": "hmmt25",
    "hmmt25_feb": "hmmt25",
    "hmmt_2025_nov": "hmmt25_nov",
    "hmmt-2025-nov": "hmmt25_nov",
    "hmmt25_nov": "hmmt25_nov",
    "longcot": "longcot_mini",
    "longcot-mini": "longcot_mini",
    "longcot_mini": "longcot_mini",
}


def list_supported_benchmarks() -> List[str]:
    return sorted(_BENCHMARK_SPECS.keys())


def canonicalize_benchmark_name(name: str) -> str:
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            "Unsupported benchmark '{}'. Supported values: {}".format(
                name, ", ".join(list_supported_benchmarks())
            )
        )
    return _ALIASES[key]


def load_benchmark(
    benchmark: str,
    dataset_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    split: Optional[str] = None,
    max_problems: Optional[int] = None,
) -> List[MathProblem]:
    canonical_name = canonicalize_benchmark_name(benchmark)
    spec = _BENCHMARK_SPECS[canonical_name]

    local_path = _resolve_local_dataset_path(canonical_name, dataset_path, dataset_root)
    records: Iterable[Mapping[str, Any]]
    active_split = split or spec.default_split

    if local_path is not None:
        records = _load_local_records(local_path)
    else:
        records = _load_hf_records(spec=spec, split=active_split)

    problems = _normalize_records(canonical_name, records)
    if max_problems is not None:
        problems = problems[:max_problems]
    return problems


def _resolve_local_dataset_path(
    benchmark: str,
    dataset_path: Optional[str],
    dataset_root: Optional[str],
) -> Optional[Path]:
    if dataset_path:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("Dataset path does not exist: {}".format(path))
        return path

    if not dataset_root:
        return None

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))

    candidates = [
        root / "{}.jsonl".format(benchmark),
        root / "{}.json".format(benchmark),
        root / "{}.jsonl".format(benchmark.replace("_", "-")),
        root / "{}.json".format(benchmark.replace("_", "-")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_local_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return payload
        raise ValueError("Expected a JSON list in {}".format(path))
    raise ValueError(
        "Unsupported local dataset format '{}'. Expected .json or .jsonl".format(path)
    )


def _load_hf_records(spec: BenchmarkSpec, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for Hugging Face benchmark loading. "
            "Install it or pass --dataset-path/--dataset-root."
        ) from exc

    if spec.hf_repo is None:
        raise ValueError("No Hugging Face dataset is configured for {}".format(spec.canonical_name))

    if spec.hf_subset is None:
        dataset = load_dataset(spec.hf_repo, split=split)
    else:
        dataset = load_dataset(spec.hf_repo, spec.hf_subset, split=split)
    return [dict(row) for row in dataset]


def _normalize_records(benchmark: str, records: Iterable[Mapping[str, Any]]) -> List[MathProblem]:
    normalized: List[MathProblem] = []
    for idx, raw in enumerate(records):
        item = dict(raw)
        if benchmark == "aime24":
            normalized.append(_normalize_aime24(item, idx))
        elif benchmark in ("aime25", "aime26"):
            normalized.append(_normalize_aime_problem(item, idx, benchmark))
        elif benchmark in ("hmmt24", "hmmt25", "hmmt25_nov"):
            normalized.append(_normalize_hmmt(item, idx, benchmark))
        elif benchmark == "longcot_mini":
            normalized.append(_normalize_longcot_mini(item, idx))
        else:
            raise ValueError("Unsupported benchmark normalization: {}".format(benchmark))
    return normalized


def _normalize_aime24(item: Dict[str, Any], idx: int) -> MathProblem:
    question = _string_or_none(item.get("problem")) or _string_or_none(item.get("question")) or ""
    answer = _extract_boxed_answer(_string_or_none(item.get("solution")))
    question_id = _string_or_none(item.get("id")) or "aime24_{:03d}".format(idx + 1)
    return MathProblem(
        benchmark="aime24",
        question_id=question_id,
        question=question,
        answer=answer,
        raw=item,
    )


def _normalize_aime_problem(item: Dict[str, Any], idx: int, benchmark: str) -> MathProblem:
    question = _string_or_none(item.get("problem")) or _string_or_none(item.get("question")) or ""
    answer = _string_or_none(item.get("answer"))
    question_id = _string_or_none(item.get("id")) or "{}_{:03d}".format(benchmark, idx + 1)
    return MathProblem(
        benchmark=benchmark,
        question_id=question_id,
        question=question,
        answer=answer,
        raw=item,
    )


def _normalize_hmmt(item: Dict[str, Any], idx: int, benchmark: str) -> MathProblem:
    question = _string_or_none(item.get("problem")) or _string_or_none(item.get("question")) or ""
    answer = _string_or_none(item.get("answer"))
    problem_idx = item.get("problem_idx", idx + 1)
    question_id = _string_or_none(item.get("id"))
    if question_id is None:
        try:
            question_id = "{}_{:03d}".format(benchmark, int(problem_idx))
        except (TypeError, ValueError):
            question_id = "{}_{}".format(benchmark, _safe_question_suffix(problem_idx, idx + 1))
    return MathProblem(
        benchmark=benchmark,
        question_id=question_id,
        question=question,
        answer=answer,
        raw=item,
    )


def _normalize_longcot_mini(item: Dict[str, Any], idx: int) -> MathProblem:
    question = _string_or_none(item.get("prompt")) or _string_or_none(item.get("question")) or ""
    answer = _string_or_none(item.get("answer"))
    question_id = _string_or_none(item.get("question_id")) or "longcot_{:03d}".format(idx + 1)
    return MathProblem(
        benchmark="longcot_mini",
        question_id=question_id,
        question=question,
        answer=answer,
        raw=item,
    )


def _extract_boxed_answer(solution: Optional[str]) -> Optional[str]:
    if solution is None:
        return None

    boxed = extract_boxed_answer(solution)
    if boxed is not None:
        return boxed

    match = re.search(r"\\boxed\s*([^\s]+)", solution)
    if match:
        return match.group(1).strip()
    return solution.strip() or None


def _safe_question_suffix(value: Any, fallback_idx: int) -> str:
    suffix = _string_or_none(value)
    if suffix is None:
        return "{:03d}".format(fallback_idx)
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix).strip("_")
    return normalized or "{:03d}".format(fallback_idx)


def _extract_prompt_content(prompt_messages: Any) -> str:
    if not isinstance(prompt_messages, list):
        return _string_or_none(prompt_messages) or ""
    for message in prompt_messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _string_or_none(message.get("content")) or ""
    if prompt_messages:
        first = prompt_messages[0]
        if isinstance(first, Mapping):
            return _string_or_none(first.get("content")) or ""
    return ""


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
