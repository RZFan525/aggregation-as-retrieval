import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.token_usage_common import summarize_jsonl_tree
from analysis.analyze_agentic_tokens import agentic_record_text
from analysis.analyze_prompted_tokens import prompted_record_text


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record) for record in records)
    path.write_text(payload + "\n", encoding="utf-8")


def test_agentic_summary_averages_message_content_tokens_by_model_and_benchmark(tmp_path):
    root = tmp_path / "Aggent_Aggregation"
    _write_jsonl(
        root / "qwen3.5-2b" / "aime24_agentic_n8.jsonl",
        [
            {
                "benchmark": "aime24",
                "aggregation_messages": [
                    {"role": "user", "content": "one two"},
                    {"role": "assistant", "content": None},
                    {"role": "tool", "content": "three"},
                ],
            },
            {
                "benchmark": "aime24",
                "aggregation_messages": [
                    {"role": "user", "content": "four five six seven"},
                ],
            },
        ],
    )

    summaries = summarize_jsonl_tree(
        root=root,
        text_builder=agentic_record_text,
        tokenizer_loader=lambda model: WhitespaceTokenizer(),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model == "qwen3.5-2b"
    assert summary.benchmark == "aime24"
    assert summary.n_records == 2
    assert summary.total_tokens == 7
    assert summary.avg_tokens == 3.5


def test_prompted_summary_averages_prompt_and_response_tokens_by_model_and_benchmark(tmp_path):
    root = tmp_path / "Prompted_Aggregation"
    _write_jsonl(
        root / "qwen3.5-9b" / "qwen3.5-9b_hmmt25_n8.jsonl",
        [
            {
                "benchmark": "hmmt25",
                "aggregation_prompt": "alpha beta",
                "aggregated_response": "gamma",
            },
            {
                "benchmark": "hmmt25",
                "aggregation_prompt": "delta",
                "aggregated_response": "epsilon zeta eta",
            },
        ],
    )

    summaries = summarize_jsonl_tree(
        root=root,
        text_builder=prompted_record_text,
        tokenizer_loader=lambda model: WhitespaceTokenizer(),
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.model == "qwen3.5-9b"
    assert summary.benchmark == "hmmt25"
    assert summary.n_records == 2
    assert summary.total_tokens == 7
    assert summary.avg_tokens == 3.5
