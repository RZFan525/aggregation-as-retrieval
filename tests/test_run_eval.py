import sys
from types import SimpleNamespace

from aggmath.evaluation.grader import grade_prediction
from aggmath.evaluation.run_eval import _evaluate_file, _extract_pred_text, _print_table
from aggmath.rollout.qwen_api import (
    extract_boxed_answer as qwen_extract_boxed_answer,
    parse_chat_completion,
)
from aggmath.utils.answer_extraction import extract_boxed_answer


def test_qwen_api_uses_shared_boxed_answer_extractor():
    assert qwen_extract_boxed_answer is extract_boxed_answer


def test_grade_prediction_disables_math_verify_timeouts(monkeypatch):
    calls = {"parse": [], "verify": []}

    def fake_parse(text, **kwargs):
        calls["parse"].append((text, kwargs))
        if "parsing_timeout" not in kwargs or kwargs.get("parsing_timeout") is not None:
            raise ValueError("parse cannot run in worker threads")
        return [text]

    def fake_verify(gold, pred, **kwargs):
        calls["verify"].append((gold, pred, kwargs))
        if "timeout_seconds" not in kwargs or kwargs.get("timeout_seconds") is not None:
            raise ValueError("verify cannot run in worker threads")
        return gold == pred

    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(parse=fake_parse, verify=fake_verify))

    assert grade_prediction("178", "\\boxed{178}") is True
    assert all(kwargs["parsing_timeout"] is None for _, kwargs in calls["parse"])
    assert calls["verify"][0][2]["timeout_seconds"] is None


def test_parse_chat_completion_can_exclude_reasoning_content():
    payload = {
        "choices": [
            {
                "message": {
                    "reasoning": "Thinking Process: internal trace",
                    "content": "\n\n{\"Summary\": \"final\"}",
                },
                "finish_reason": "stop",
            }
        ]
    }

    with_reasoning = parse_chat_completion(payload)
    without_reasoning = parse_chat_completion(payload, include_reasoning=False)

    assert with_reasoning.choices[0].text.startswith("Thinking Process:")
    assert without_reasoning.choices[0].text == "{\"Summary\": \"final\"}"


def test_extract_pred_text_prefers_stored_answer_over_noisy_reasoning():
    rollout = {
        "reasoning": (
            "A long solution with an earlier bad box: \\boxed{999}\n"
            "Then the final answer is \\boxed{204}.\n"
            "\\boxed{204}\n"
            "\\boxed{2"
        ),
        "answer": "204",
    }

    assert _extract_pred_text(rollout) == "\\boxed{204}"


def test_extract_pred_text_uses_last_complete_box_when_answer_missing():
    rollout = {
        "reasoning": (
            "An intermediate value is \\boxed{12}.\n"
            "The final answer is \\boxed{45}.\n"
            "\\boxed{"
        ),
    }

    assert _extract_pred_text(rollout) == "\\boxed{45}"


def test_print_table_includes_model_directory_for_math_rollouts(capsys):
    summaries = [
        {
            "file": "outputs/math_rollout/qwen3.5-2b/aime24_thinking_n8.jsonl",
            "n_problems": 30,
            "n_rollouts_per_problem": 8,
            "pass_at_k": {"pass@1": 0.25, "pass@8": 0.5},
        }
    ]

    _print_table(summaries, [1, 8])

    output = capsys.readouterr().out
    assert "qwen3.5-2b/aime24_thinking_n8.jsonl" in output


def test_eval_output_names_include_model_when_using_shared_output_dir(tmp_path):
    output_dir = tmp_path / "eval"
    paths = [
        tmp_path
        / "outputs"
        / "math_rollout"
        / "qwen3.5-2b"
        / "aime24_thinking_n8.jsonl",
        tmp_path
        / "outputs"
        / "math_rollout"
        / "qwen3.5-9b"
        / "aime24_thinking_n8.jsonl",
    ]
    for idx, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            (
                '{"problem_id": "%s", "instance": {"answer": "1"}, '
                '"rollouts": []}\n'
            )
            % idx,
            encoding="utf-8",
        )

    summaries = [
        _evaluate_file(
            path=path,
            k_values=[1],
            output_dir=output_dir,
            write_per_problem=False,
        )
        for path in paths
    ]

    summary_paths = [summary["_summary_path"] for summary in summaries]
    assert len(set(summary_paths)) == 2
    assert (output_dir / "qwen3.5-2b_aime24_thinking_n8.eval.json").exists()
    assert (output_dir / "qwen3.5-9b_aime24_thinking_n8.eval.json").exists()
