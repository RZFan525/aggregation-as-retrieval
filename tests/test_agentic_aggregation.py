import json
from pathlib import Path

from aggmath.aggregation import evaluation
from aggmath.aggregation import run_agentic_aggregation as pipeline


class FakeAgent:
    def __init__(self):
        self.calls = []

    def _run(self, question, run_results):
        self.calls.append((question, run_results))
        return {
            "result": {
                "solution": "We inspect the candidates and conclude \\boxed{2}.",
                "reason": "The first candidate is consistent.",
            },
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "TASK:\nWhat is 1+1?\n\n"
                        "SOLUTION METADATA:\n"
                        "Solution 0: 1 steps, 6 tokens, answer: 2\n\n"
                        "You have 1 candidate solutions."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "I should inspect the candidate answers before finishing.",
                    "tool_calls": [
                        {
                            "id": "answers-1",
                            "type": "function",
                            "function": {
                                "name": "get_answer",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "answers-1",
                    "name": "get_answer",
                    "content": json.dumps(
                        [{"solution_id": 0, "answer": "2", "steps": 1, "tokens": 6}]
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "The tool observation supports the first answer.",
                    "tool_calls": [
                        {
                            "id": "finish-1",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {
                                        "solution": "We conclude \\boxed{2}.",
                                        "reason": "checked",
                                    }
                                ),
                            },
                        }
                    ],
                },
            ],
            "stats": {"iterations": 2, "tool_calls": {"get_answer": 1, "finish": 1}},
        }


def test_run_agentic_aggregation_writes_records_with_agentic_trace(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "benchmark": "aime24",
                "problem_id": "p1",
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
    agent = FakeAgent()

    result = pipeline.run_agentic_aggregation(
        input_jsonl=input_path,
        output_dir=tmp_path / "out",
        model="qwen3.5-9b",
        agent=agent,
        k=1,
        overwrite=True,
        max_workers=1,
    )

    assert result.output_path == tmp_path / "out" / "qwen3.5-9b" / "aime24_agentic_n1.jsonl"
    assert agent.calls == [
        (
            "What is 1+1?",
            [{"rollout_idx": 0, "reasoning": "It is \\boxed{2}.", "answer": "2"}],
        )
    ]

    records = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["benchmark"] == "aime24"
    assert record["problem_id"] == "p1"
    assert record["gold_answer"] == "2"
    assert record["aggregation_model"] == "Qwen/Qwen3.5-9B"
    assert record["summary_model"] == "Qwen/Qwen3.5-9B"
    assert record["aggregated_response"] == "We inspect the candidates and conclude \\boxed{2}."
    assert record["aggregated_answer"] == "2"
    assert record["aggregation_reason"] == "The first candidate is consistent."
    assert record["solution_metadata"] == "Solution 0: 1 steps, 6 tokens, answer: 2"
    assert record["aggregation_stats"]["tool_calls"]["finish"] == 1
    assert record["aggregation_messages"][1]["role"] == "user"
    assert record["aggregation_messages"][2]["reasoning"] == (
        "I should inspect the candidate answers before finishing."
    )
    assert record["aggregation_messages"][3]["role"] == "tool"
    assert record["aggregation_messages"][3]["name"] == "get_answer"
    assert "solution_id" in record["aggregation_messages"][3]["content"]


def test_evaluation_module_reports_accuracy_missing_and_context_limits(tmp_path, monkeypatch):
    input_path = tmp_path / "qwen3.5-9b_aime24_agentic_n1.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_id": "1",
                        "gold_answer": "2",
                        "aggregated_response": "Final answer is \\boxed{2}.",
                        "aggregated_answer": "2",
                        "aggregation_stats": {"context_limit_reached": False},
                    }
                ),
                json.dumps(
                    {
                        "problem_id": "2",
                        "gold_answer": "5",
                        "aggregated_answer": None,
                        "aggregated_response": None,
                        "aggregation_stats": {
                            "context_limit_reached": True,
                            "tool_call_errors": 2,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluation,
        "grade_prediction",
        lambda gold, pred: gold == "2" and pred == "\\boxed{2}",
    )

    summary = evaluation.evaluate_agentic_aggregation_file(
        input_path,
        output_dir=tmp_path / "eval",
        write_per_problem=True,
    )

    assert summary["n_total"] == 2
    assert summary["n_correct"] == 1
    assert summary["n_missing_answer"] == 1
    assert summary["n_empty_response"] == 1
    assert summary["n_context_limit_reached"] == 1
    assert summary["n_tool_call_errors"] == 2
    assert summary["accuracy"] == 0.5
    assert Path(summary["_summary_path"]).exists()
    assert (tmp_path / "eval" / "qwen3.5-9b_aime24_agentic_n1.eval.jsonl").exists()


def test_build_litellm_kwargs_defaults_summary_to_aggregation_endpoint():
    kwargs = pipeline.build_litellm_kwargs(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        thinking=True,
        request_timeout=123,
        max_retries=4,
    )

    assert kwargs["model"] == "hosted_vllm/Qwen/Qwen3.5-9B"
    assert kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert kwargs["timeout"] == 123
    assert kwargs["num_retries"] == 4


def test_build_agent_keeps_summary_generation_non_thinking():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    agent = pipeline.build_agent(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:8000/v1",
        summary_model="qwen3.5-9b",
        max_tokens=1024,
        summary_max_tokens=512,
        thinking=True,
        tokenizer=FakeTokenizer(),
    )

    assert (
        agent.llm_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is True
    )
    summary_tool = agent.tool_map["get_solution_summary"]
    assert (
        summary_tool.llm_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        is False
    )


def test_build_agent_can_select_bm25_global_search_tool():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    agent = pipeline.build_agent(
        model="qwen3.5-9b",
        base_url="http://127.0.0.1:8000/v1",
        tokenizer=FakeTokenizer(),
        search_tool="bm25",
    )

    search_tool = agent.tool_map["search_solution"]
    search_schema = search_tool.get_tool_definitions()
    assert search_tool.__class__.__name__ == "SearchAllSolutionsBM25Tool"
    assert "solution_id" not in search_schema["function"]["parameters"]["properties"]
    assert "BM25" in search_schema["function"]["description"]
