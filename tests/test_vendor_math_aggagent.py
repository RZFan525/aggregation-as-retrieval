import json
import sys
import types
from importlib import util
from pathlib import Path

import pytest


class _FakeRougeScore:
    def __init__(self, recall):
        self.recall = recall


class _FakeRougeScorer:
    def __init__(self, *args, **kwargs):
        pass

    def score(self, query, text):
        query_terms = set(query.lower().replace(".", "").split())
        text_terms = set(text.lower().replace(".", "").split())
        recall = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
        return {"rougeL": _FakeRougeScore(recall)}


rouge_pkg = types.ModuleType("rouge_score")
rouge_scorer_mod = types.ModuleType("rouge_score.rouge_scorer")
rouge_scorer_mod.RougeScorer = _FakeRougeScorer
rouge_pkg.rouge_scorer = rouge_scorer_mod
sys.modules.setdefault("rouge_score", rouge_pkg)
sys.modules.setdefault("rouge_score.rouge_scorer", rouge_scorer_mod)

litellm_mod = types.ModuleType("litellm")
litellm_mod.BadRequestError = RuntimeError
sys.modules.setdefault("litellm", litellm_mod)


class _FakeBM25:
    def __init__(self, corpus=None, **kwargs):
        self.corpus = corpus or []
        self.kwargs = kwargs
        self.corpus_tokens = []

    def index(self, corpus_tokens):
        self.corpus_tokens = corpus_tokens

    def retrieve(self, query_tokens, k=5):
        query = set(query_tokens[0])
        scored = []
        for idx, tokens in enumerate(self.corpus_tokens):
            score = len(query & set(tokens))
            if score > 0:
                scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        docs = [[self.corpus[idx] for score, idx in scored[:k]]]
        scores = [[float(score) for score, idx in scored[:k]]]
        return docs, scores


def _fake_bm25_tokenize(value, **kwargs):
    del kwargs
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return [
        [
            token.strip("\\{}.,;:()[]").lower()
            for token in text.split()
            if token.strip("\\{}.,;:()[]")
        ]
        for text in values
    ]


bm25s_mod = types.ModuleType("bm25s")
bm25s_mod.BM25 = _FakeBM25
bm25s_mod.tokenize = _fake_bm25_tokenize
sys.modules.setdefault("bm25s", bm25s_mod)

_TOOLS_PATH = Path(__file__).resolve().parents[1] / "vendor" / "aggagent_core" / "tools.py"
_TOOLS_SPEC = util.spec_from_file_location("aggagent_core_tools_under_test", _TOOLS_PATH)
tools = util.module_from_spec(_TOOLS_SPEC)
assert _TOOLS_SPEC.loader is not None
_TOOLS_SPEC.loader.exec_module(tools)

FinishTool = tools.FinishTool
GetAnswerTool = tools.GetAnswerTool
GetReasoningStepTool = tools.GetReasoningStepTool
GetSolutionSummaryTool = tools.GetSolutionSummaryTool
SearchSolutionTool = tools.SearchSolutionTool
SearchAllSolutionsBM25Tool = tools.SearchAllSolutionsBM25Tool
build_solution_environment = tools.build_solution_environment
format_metadata = tools.format_metadata


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_solution_environment_chunks_by_qwen_token_budget_without_cutting_sentences():
    reasoning = (
        "alpha beta gamma. delta epsilon.\n\n"
        "zeta eta theta. iota kappa lambda."
    )

    solutions = build_solution_environment(
        [{"reasoning": reasoning, "answer": "42"}],
        tokenizer=FakeTokenizer(),
        step_token_target=5,
    )

    assert [step["content"] for step in solutions[0]["steps"]] == [
        "alpha beta gamma. delta epsilon.",
        "zeta eta theta.",
        "iota kappa lambda.",
    ]
    assert [step["step"] for step in solutions[0]["steps"]] == [0, 1, 2]
    assert format_metadata(solutions) == "Solution 0: 3 steps, 11 tokens, answer: 42"


def test_solution_environment_reads_grouped_rollout_records_without_reextracting_answer():
    record = {
        "problem_id": "aime24-1",
        "question": "What is 1+1?",
        "rollouts": [
            {
                "rollout_idx": 0,
                "reasoning": "Compute directly. The result is \\boxed{2}.",
                "answer": "2",
            },
            {
                "rollout_idx": 1,
                "reasoning": "This text has a boxed value \\boxed{999}, but answer extraction failed upstream.",
                "answer": None,
            },
        ],
    }

    solutions = build_solution_environment(
        record,
        tokenizer=FakeTokenizer(),
        step_token_target=100,
    )

    assert [solution["solution_id"] for solution in solutions] == [0, 1]
    assert [solution["answer"] for solution in solutions] == ["2", None]
    assert "Compute directly" in solutions[0]["reasoning"]
    assert "boxed value" in solutions[1]["reasoning"]
    assert format_metadata(solutions) == (
        "Solution 0: 1 steps, 6 tokens, answer: 2\n\n"
        "Solution 1: 1 steps, 12 tokens, answer: None"
    )


def test_math_tools_use_solution_ids_and_search_reasoning_steps_only():
    solutions = build_solution_environment(
        [
            {
                "reasoning": "First derive x. Then compute theta carefully. Therefore \\boxed{42}.",
                "answer": "42",
            },
            {
                "reasoning": "A different attempt gives \\boxed{7}.",
                "answer": "7",
            },
        ],
        tokenizer=FakeTokenizer(),
        step_token_target=4,
    )

    answer_schema = GetAnswerTool().get_tool_definitions()
    assert answer_schema["function"]["name"] == "get_answer"
    assert answer_schema["function"]["parameters"]["properties"] == {}
    assert answer_schema["function"]["parameters"]["required"] == []

    answers = GetAnswerTool().call({"solution_id": 0}, solutions=solutions)
    assert answers == [
        {"solution_id": 0, "answer": "42", "steps": 3, "tokens": 9},
        {"solution_id": 1, "answer": "7", "steps": 1, "tokens": 5},
    ]

    search_schema = SearchSolutionTool().get_tool_definitions()
    assert "role" not in search_schema["function"]["parameters"]["properties"]

    matches = SearchSolutionTool().call(
        {"solution_id": 0, "query": "theta", "k": 2},
        solutions=solutions,
    )
    assert matches[0]["solution_id"] == 0
    assert matches[0]["step"] == 1
    assert "theta" in matches[0]["content"]

    segment = GetReasoningStepTool().call(
        {"solution_id": 0, "start_step": 1, "end_step": 10},
        solutions=solutions,
    )
    assert [entry["step"] for entry in segment] == [1, 2]


def test_bm25_search_solution_searches_all_steps_without_solution_id():
    solutions = build_solution_environment(
        [
            {
                "reasoning": "We try a parity argument. It does not use the quadratic condition.",
                "answer": "1",
            },
            {
                "reasoning": "The discriminant of the quadratic is zero, so the double root condition applies.",
                "answer": "2",
            },
        ],
        tokenizer=FakeTokenizer(),
        step_token_target=100,
    )

    schema = SearchAllSolutionsBM25Tool().get_tool_definitions()
    properties = schema["function"]["parameters"]["properties"]
    assert schema["function"]["name"] == "search_solution"
    assert "solution_id" not in properties
    assert schema["function"]["parameters"]["required"] == ["query"]

    matches = SearchAllSolutionsBM25Tool().call(
        {"query": "discriminant quadratic double root", "k": 2},
        solutions=solutions,
        tool_state={},
    )

    assert matches[0]["source"] == "Solution 1, step 0"
    assert matches[0]["solution_id"] == 1
    assert matches[0]["step"] == 0
    assert matches[0]["content"].startswith("[Solution 1, step 0]\n")
    assert "discriminant" in matches[0]["content"]


def test_solution_summary_tool_calls_model_with_full_solution_reasoning():
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = types.SimpleNamespace(
            content="Reasoning: Keep the derivation and final claim.\nSummary: Derives the value and claims 42."
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    solutions = build_solution_environment(
        [{"reasoning": "Full derivation line one.\nFull derivation line two.", "answer": "42"}],
        tokenizer=FakeTokenizer(),
        step_token_target=100,
    )

    result = GetSolutionSummaryTool(
        model="qwen3.5-9b",
        api_base="http://localhost:8000/v1",
        completion_fn=fake_completion,
    ).call({"solution_id": 0}, solutions=solutions, question="What is x?")

    assert result == {
        "solution_id": 0,
        "answer": "42",
        "summary": "Derives the value and claims 42.",
    }
    assert calls[0]["model"] == "hosted_vllm/qwen3.5-9b"
    assert calls[0]["api_base"] == "http://localhost:8000/v1"
    assert "tools" not in calls[0]
    assert "Full candidate reasoning:\nFull derivation line one." in calls[0]["messages"][0]["content"]


def test_solution_summary_tool_reads_reasoning_content_when_content_is_empty():
    def fake_completion(**kwargs):
        del kwargs
        message = types.SimpleNamespace(
            content="",
            reasoning_content="Reasoning: Content is empty, so read reasoning content.\nSummary: Uses a valid equation and claims 42.",
        )
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    solutions = build_solution_environment(
        [{"reasoning": "Full derivation.", "answer": "42"}],
        tokenizer=FakeTokenizer(),
        step_token_target=100,
    )

    result = GetSolutionSummaryTool(
        model="qwen3.5-9b",
        api_base="http://localhost:8000/v1",
        completion_fn=fake_completion,
    ).call({"solution_id": 0}, solutions=solutions, question="What is x?")

    assert result["summary"] == "Uses a valid equation and claims 42."


def test_finish_requires_boxed_math_answer():
    tool = FinishTool(model="qwen3.5-9b")

    accepted = tool.call(
        {"solution": "We solve step by step and get \\boxed{42}.", "reason": "checked"}
    )
    assert accepted == {
        "solution": "We solve step by step and get \\boxed{42}.",
        "reason": "checked",
    }

    rejected = tool.call({"solution": "The answer is 42.", "reason": "missing box"})
    assert "boxed" in rejected


def test_agent_prepares_math_solution_tools_with_zero_based_metadata():
    from vendor.aggagent_core.agent import AggAgent

    agent = AggAgent(
        model="qwen3.5-9b",
        tokenizer=FakeTokenizer(),
        solution_step_tokens=5,
    )

    assert set(agent.tool_map) == {
        "get_answer",
        "search_solution",
        "get_reasoning_step",
        "get_solution_summary",
        "finish",
    }

    solutions = agent._prepare_solutions(
        {
            "rollouts": [
                {"reasoning": "one two three. four five six.", "answer": "6"}
            ]
        }
    )
    assert format_metadata(solutions) == "Solution 0: 2 steps, 6 tokens, answer: 6"


def test_agent_trace_keeps_finish_tool_observation():
    from vendor.aggagent_core.agent import AggAgent

    class FakeFunction:
        name = "finish"
        arguments = json.dumps(
            {
                "solution": "A complete solution ends with \\boxed{2}.",
                "reason": "verified",
            }
        )

    class FakeToolCall:
        id = "finish-1"
        function = FakeFunction()

    class FakeMessage(dict):
        def __init__(self):
            super().__init__(
                content="",
                reasoning_content="I verified the candidate and will finish.",
            )
            self.tool_calls = [FakeToolCall()]

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

    agent = AggAgent(
        model="qwen3.5-9b",
        tokenizer=FakeTokenizer(),
        solution_step_tokens=100,
    )
    agent.call_server = lambda *args, **kwargs: FakeResponse()

    output = agent._run(
        "What is 1+1?",
        [{"reasoning": "Compute directly. It is \\boxed{2}.", "answer": "2"}],
    )

    assert output["result"]["solution"] == "A complete solution ends with \\boxed{2}."
    assert output["messages"][-2]["role"] == "assistant"
    assert output["messages"][-2]["reasoning"] == "I verified the candidate and will finish."
    assert output["messages"][-1]["role"] == "tool"
    assert output["messages"][-1]["name"] == "finish"
    assert "A complete solution ends with \\\\boxed{2}." in output["messages"][-1]["content"]


def test_agent_raises_tool_infrastructure_errors():
    from vendor.aggagent_core.agent import AggAgent

    class FakeFunction:
        name = "search_solution"
        arguments = json.dumps({"query": "one plus one"})

    class FakeToolCall:
        id = "search-1"
        function = FakeFunction()

    class FakeMessage(dict):
        def __init__(self):
            super().__init__(content="", reasoning_content="")
            self.tool_calls = [FakeToolCall()]

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = None

    agent = AggAgent(
        model="qwen3.5-9b",
        tokenizer=FakeTokenizer(),
        solution_step_tokens=100,
        max_context_tokens=10**9,
    )
    agent.call_server = lambda *args, **kwargs: FakeResponse()

    def fail_tool_call(*args, **kwargs):
        del args, kwargs
        raise ImportError("bm25s is required for BM25 search_solution.")

    agent.custom_call_tool = fail_tool_call

    with pytest.raises(ImportError, match="bm25s is required"):
        agent._run(
            "What is 1+1?",
            [{"reasoning": "Compute directly. It is \\boxed{2}.", "answer": "2"}],
        )


def test_agent_keeps_non_infrastructure_tool_errors_as_observations():
    from vendor.aggagent_core.agent import AggAgent

    class SearchFunction:
        name = "search_solution"
        arguments = json.dumps({"query": "one plus one"})

    class FinishFunction:
        name = "finish"
        arguments = json.dumps(
            {
                "solution": "A complete solution ends with \\boxed{2}.",
                "reason": "verified",
            }
        )

    class FakeToolCall:
        def __init__(self, function):
            self.id = function.name
            self.function = function

    class FakeMessage(dict):
        def __init__(self, function):
            super().__init__(content="", reasoning_content="")
            self.tool_calls = [FakeToolCall(function)]

    class FakeResponse:
        usage = None

        def __init__(self, function):
            self.choices = [types.SimpleNamespace(message=FakeMessage(function))]

    responses = iter([FakeResponse(SearchFunction()), FakeResponse(FinishFunction())])
    agent = AggAgent(
        model="qwen3.5-9b",
        tokenizer=FakeTokenizer(),
        solution_step_tokens=100,
        max_context_tokens=10**9,
    )
    agent.call_server = lambda *args, **kwargs: next(responses)

    def fail_then_finish(tool_name, *args, **kwargs):
        del args, kwargs
        if tool_name == "search_solution":
            raise ValueError("transient search bug")
        return {
            "solution": "A complete solution ends with \\boxed{2}.",
            "reason": "verified",
        }

    agent.custom_call_tool = fail_then_finish

    output = agent._run(
        "What is 1+1?",
        [{"reasoning": "Compute directly. It is \\boxed{2}.", "answer": "2"}],
    )

    assert output["result"]["solution"] == "A complete solution ends with \\boxed{2}."
    assert output["stats"]["tool_call_errors"] == 1
    assert "Tool call is not a valid JSON" in output["messages"][3]["content"]
