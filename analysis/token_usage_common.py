"""Shared token-usage helpers for AggMath aggregation analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class TokenSummary:
    model: str
    benchmark: str
    n_records: int
    total_tokens: int
    avg_tokens: float


TextBuilder = Callable[[dict[str, Any]], str]
TokenizerLoader = Callable[[str], Any]


def load_qwen35_tokenizer(model: str) -> Any:
    """Load the Qwen3.5 tokenizer matching a model directory or alias."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for Qwen3.5 token counting. Run this script "
            "inside the AggMath/vLLM environment, for example "
            "`/mnt/weka/home/yongxin.wang/workspace/runze/miniconda3/envs/aggmath/bin/python "
            "analysis/analyze_agentic_tokens.py`."
        ) from exc

    from aggmath.rollout.qwen_api import normalize_model_name

    tokenizer_name = normalize_model_name(model)
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def count_text_tokens(text: str, tokenizer: Any) -> int:
    """Count text tokens with a HuggingFace-style tokenizer."""
    if not text:
        return 0
    try:
        tokenized = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokenized = tokenizer.encode(text)
    return _count_tokenized(tokenized)


def summarize_jsonl_tree(
    root: Path,
    text_builder: TextBuilder,
    tokenizer_loader: TokenizerLoader = load_qwen35_tokenizer,
) -> list[TokenSummary]:
    """Summarize average token usage for every model/benchmark JSONL group."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError("Output root does not exist: {}".format(root))

    tokenizer_cache: dict[str, Any] = {}
    grouped: dict[tuple[str, str], dict[str, int]] = {}

    for path in _iter_jsonl_files(root):
        model = path.parent.name
        if model not in tokenizer_cache:
            tokenizer_cache[model] = tokenizer_loader(model)
        tokenizer = tokenizer_cache[model]

        for record in _read_jsonl(path):
            benchmark = _benchmark_from_record_or_path(record, path, model)
            token_count = count_text_tokens(text_builder(record), tokenizer)
            key = (model, benchmark)
            stats = grouped.setdefault(key, {"n_records": 0, "total_tokens": 0})
            stats["n_records"] += 1
            stats["total_tokens"] += token_count

    summaries: list[TokenSummary] = []
    for (model, benchmark), stats in sorted(grouped.items()):
        n_records = stats["n_records"]
        total_tokens = stats["total_tokens"]
        avg_tokens = total_tokens / n_records if n_records else 0.0
        summaries.append(
            TokenSummary(
                model=model,
                benchmark=benchmark,
                n_records=n_records,
                total_tokens=total_tokens,
                avg_tokens=avg_tokens,
            )
        )
    return summaries


def print_summary_table(summaries: Iterable[TokenSummary]) -> None:
    rows = list(summaries)
    if not rows:
        print("No JSONL records found.")
        return

    headers = ["model", "benchmark", "n_records", "avg_tokens", "total_tokens"]
    table_rows = [
        [
            row.model,
            row.benchmark,
            str(row.n_records),
            "{:.2f}".format(row.avg_tokens),
            str(row.total_tokens),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows))
        for col in range(len(headers))
    ]
    print("  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))


def write_json_summary(path: Optional[Path], summaries: Iterable[TokenSummary]) -> None:
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(summary) for summary in summaries]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv_summary(path: Optional[Path], summaries: Iterable[TokenSummary]) -> None:
    if path is None:
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "benchmark", "n_records", "total_tokens", "avg_tokens"],
        )
        writer.writeheader()
        writer.writerows(rows)


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--csv-out", default=None, help="Optional path to write CSV summary.")


def _count_tokenized(tokenized: Any) -> int:
    input_ids = _extract_input_ids(tokenized)
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        shape_tuple = tuple(shape)
        return int(shape_tuple[-1]) if shape_tuple else 0
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


def _iter_jsonl_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.glob("*/*.jsonl") if path.is_file())


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed JSON in {}:{}: {}".format(path, line_idx, exc)) from exc
            if isinstance(record, dict):
                yield record


def _benchmark_from_record_or_path(record: dict[str, Any], path: Path, model: str) -> str:
    benchmark = record.get("benchmark")
    if isinstance(benchmark, str) and benchmark.strip():
        return benchmark.strip()

    stem = path.stem
    if stem.startswith(model + "_"):
        stem = stem[len(model) + 1 :]
    if "_agentic_n" in stem:
        stem = stem.split("_agentic_n", 1)[0]
    elif "_n" in stem:
        stem = stem.rsplit("_n", 1)[0]
    return stem or "unknown"
