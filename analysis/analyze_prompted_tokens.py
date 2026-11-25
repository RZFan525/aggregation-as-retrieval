"""Analyze token usage in Prompted Aggregation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.token_usage_common import (  # noqa: E402
    add_common_output_args,
    print_summary_table,
    summarize_jsonl_tree,
    write_csv_summary,
    write_json_summary,
)


def prompted_record_text(record: dict[str, Any]) -> str:
    """Return aggregation prompt and final response concatenated."""
    parts: list[str] = []
    for key in ("aggregation_prompt", "aggregated_response"):
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average Qwen3.5-token counts for aggregation_prompt plus "
            "aggregated_response in Prompted Aggregation outputs."
        )
    )
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT / "outputs" / "Prompted_Aggregation"),
        help="Root directory containing model subdirectories of prompted JSONL outputs.",
    )
    add_common_output_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = summarize_jsonl_tree(
        root=Path(args.root),
        text_builder=prompted_record_text,
    )
    print_summary_table(summaries)
    write_json_summary(Path(args.json_out) if args.json_out else None, summaries)
    write_csv_summary(Path(args.csv_out) if args.csv_out else None, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
