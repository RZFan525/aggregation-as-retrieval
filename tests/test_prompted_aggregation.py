import json
from pathlib import Path

from aggmath.rollout.qwen_api import OpenAICompatClient, default_generation_config
from baselines.prompted_aggregation import evaluate
from baselines.prompted_aggregation.run_prompted_aggregation import (
    AGGREGATION_PROMPT_TEMPLATE,
    BudgetedPrompt,
    TransformersTokenCounter,
    build_aggregation_prompt,
    build_parser,
    build_output_path,
    format_solution_attempts,
    run_prompted_aggregation,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        return list(messages[0]["content"])


class FakeDictTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        token_ids = list(range(len(messages[0]["content"])))
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}


class FakeBatchedDictTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        token_ids = list(range(len(messages[0]["content"])))
        return {"input_ids": [token_ids], "attention_mask": [[1] * len(token_ids)]}


def test_token_counter_counts_input_ids_when_chat_template_returns_dict():
    counter = TransformersTokenCounter(FakeDictTokenizer())

    assert counter.count_prompt("hello") == 5


def test_token_counter_counts_single_batched_input_ids():
    counter = TransformersTokenCounter(FakeBatchedDictTokenizer())

    assert counter.count_prompt("hello") == 5


def test_prompt_template_keeps_problem_and_solutions_placeholders():
    assert "{problem}" in AGGREGATION_PROMPT_TEMPLATE
    assert "{solutions}" in AGGREGATION_PROMPT_TEMPLATE


def test_solution_attempts_are_numbered_and_use_reasoning_text():
    rollouts = [
        {"rollout_idx": 3, "reasoning": "first solution", "answer": "1"},
        {"rollout_idx": 4, "reasoning": "second solution", "answer": "2"},
    ]

    assert format_solution_attempts(rollouts) == (
        "Solution attempt 1:\nfirst solution\n\n"
        "Solution attempt 2:\nsecond solution"
    )


def test_prompt_budget_truncates_concatenated_solutions_from_end_only():
    counter = TransformersTokenCounter(FakeTokenizer())
    rollouts = [
        {"reasoning": "A" * 40},
        {"reasoning": "B" * 40},
    ]

    budgeted = build_aggregation_prompt(
        problem="short problem",
        rollouts=rollouts,
        k=2,
        max_prompt_tokens=counter.count_prompt(
            AGGREGATION_PROMPT_TEMPLATE.format(
                problem="short problem",
                solutions="Solution attempt 1:\n" + ("A" * 40) + "\n\nSolution attempt 2:\n",
            )
        ),
        token_counter=counter,
    )

    assert isinstance(budgeted, BudgetedPrompt)
    assert budgeted.was_truncated is True
    assert "short problem" in budgeted.prompt
    assert "A" * 40 in budgeted.prompt
    assert "B" not in budgeted.prompt


def test_build_output_path_uses_model_benchmark_and_k(tmp_path):
    path = build_output_path(
        output_dir=tmp_path,
        model="Qwen/Qwen3.5-9B",
        benchmark="aime24",
        k=8,
    )

    assert path == tmp_path / "qwen3.5-9b_aime24_n8.jsonl"


def test_run_prompted_aggregation_writes_expected_records(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "benchmark": "aime24",
                "problem_id": "67",
                "question": "What is 1+1?",
                "instance": {"answer": "2"},
                "rollouts": [
                    {"rollout_idx": 0, "reasoning": "It is \\boxed{2}.", "answer": "2"},
                    {"rollout_idx": 1, "reasoning": "It is \\boxed{3}.", "answer": "3"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sent_prompts = []

    class FakeClient:
        def chat_completion(self, **kwargs):
            assert kwargs["config"].top_k is None
            sent_prompts.append(kwargs["messages"][0]["content"])

            class Choice:
                text = "Final answer is \\boxed{2}."
                finish_reason = "stop"

            class Parsed:
                choices = [Choice()]
                usage = {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
                raw_response = {"ok": True}

            return Parsed()

    output_path = run_prompted_aggregation(
        input_jsonl=input_path,
        output_dir=tmp_path / "out",
        model="qwen3.5-9b",
        client=FakeClient(),
        k=2,
        max_model_len=2048,
        max_tokens=128,
        prompt_token_buffer=32,
        token_counter=TransformersTokenCounter(FakeTokenizer()),
        overwrite=True,
        max_workers=1,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["benchmark"] == "aime24"
    assert record["problem_id"] == "67"
    assert record["question"] == "What is 1+1?"
    assert record["gold_answer"] == "2"
    assert record["rollouts"][0]["reasoning"] == "It is \\boxed{2}."
    assert record["aggregation_prompt"] == sent_prompts[0]
    assert record["aggregated_response"] == "Final answer is \\boxed{2}."
    assert record["aggregated_answer"] == "2"
    assert record["prompt_truncated"] is False
    assert record["token_usage"]["prompt_tokens"] == 10
    assert record["token_usage"]["completion_tokens"] == 5
    assert record["token_usage"]["total_tokens"] == 15
    assert record["token_usage"]["estimated_prompt_tokens"] > 0
    assert record["token_usage"]["original_estimated_prompt_tokens"] > 0
    assert record["token_usage"]["max_prompt_tokens"] == 1888


def test_parser_defaults_to_low_concurrency_and_has_no_top_k():
    parser = build_parser()
    args = parser.parse_args(["--input-jsonl", "x.jsonl"])

    assert args.max_workers == 2
    assert args.request_timeout == 300
    assert not hasattr(args, "top_k")


def test_openai_compat_payload_omits_top_k_when_config_has_none():
    client = OpenAICompatClient(base_url="http://localhost:8000/v1")
    config = default_generation_config(thinking=True, max_tokens=32, top_k=None)

    payload = client._build_payload(
        model="qwen",
        messages=[{"role": "user", "content": "hello"}],
        config=config,
        n=1,
    )

    assert "top_k" not in payload["extra_body"]


def test_openai_compat_payload_omits_chat_template_kwargs_when_thinking_is_none():
    client = OpenAICompatClient(base_url="http://localhost:8001/v1")
    config = default_generation_config(thinking=None, max_tokens=32, top_k=20)

    payload = client._build_payload(
        model="qwen",
        messages=[{"role": "user", "content": "hello"}],
        config=config,
        n=1,
    )

    assert "chat_template_kwargs" not in payload["extra_body"]
    assert payload["extra_body"]["top_k"] == 20


def test_evaluate_prompted_aggregation_outputs_summary(tmp_path, monkeypatch):
    input_path = tmp_path / "qwen3.5-9b_aime24_n8.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_id": "1",
                        "gold_answer": "2",
                        "aggregated_answer": "2",
                    }
                ),
                json.dumps(
                    {
                        "problem_id": "2",
                        "gold_answer": "5",
                        "aggregated_answer": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate,
        "grade_prediction",
        lambda gold, pred: gold == "2" and pred == "\\boxed{2}",
    )

    summary = evaluate.evaluate_file(
        input_path,
        output_dir=tmp_path / "eval",
        write_per_problem=True,
    )

    assert summary["n_total"] == 2
    assert summary["n_correct"] == 1
    assert summary["n_wrong"] == 1
    assert summary["n_missing_answer"] == 1
    assert summary["accuracy"] == 0.5
    assert Path(summary["_summary_path"]).exists()
    assert (tmp_path / "eval" / "qwen3.5-9b_aime24_n8.eval.jsonl").exists()
