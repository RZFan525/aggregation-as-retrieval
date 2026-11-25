import json

import numpy as np
import pytest

from baselines.prompted_aggregation import evaluate
from baselines.prompted_aggregation.run_prompted_aggregation import TransformersTokenCounter
from baselines.single_shot_rag.embed_passages import collect_corpus
from baselines.single_shot_rag.passages import (
    bm25_rank,
    build_passages,
    dense_rank,
    format_query,
    format_retrieved_passages,
    passage_key,
)
from baselines.single_shot_rag.run_single_shot_rag import load_embeddings, retrieve, run_single_shot_rag


class FakeWordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return list(messages[0]["content"])


ROLLOUTS = [
    {
        "rollout_idx": 0,
        "reasoning": "Use modular arithmetic on the residue.\nThe residue modulo nine is four.\nSo the answer is \\boxed{4}.",
        "answer": "4",
    },
    {
        "rollout_idx": 1,
        "reasoning": "Try small cases first.\nThe pattern repeats every three terms.\nThus \\boxed{7}.",
        "answer": "7",
    },
]


def _passages():
    return build_passages(ROLLOUTS, k=2, tokenizer=FakeWordTokenizer(), step_token_target=8)


def test_build_passages_uses_aggagent_steps_for_each_rollout():
    passages = _passages()
    keys = [passage_key(p) for p in passages]
    assert len(keys) == len(set(keys))
    assert {solution_id for solution_id, _ in keys} == {0, 1}
    assert all(p["tokens"] <= 8 for p in passages)
    joined = " ".join(p["content"] for p in passages)
    assert "residue modulo nine" in joined and "pattern repeats" in joined


def test_build_passages_requires_k_rollouts():
    with pytest.raises(ValueError):
        build_passages(ROLLOUTS, k=3, tokenizer=FakeWordTokenizer())


def test_bm25_rank_prefers_matching_passages_and_caps_at_top_m():
    passages = _passages()
    ranked = bm25_rank("residue modulo nine", passages, top_m=2)
    assert 1 <= len(ranked) <= 2
    assert passages[ranked[0][0]]["solution_id"] == 0
    assert ranked[0][1] >= ranked[-1][1]


def test_dense_rank_orders_by_cosine():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    ranked = dense_rank(np.asarray([0.0, 1.0], dtype=np.float32), vectors, top_m=2)
    assert [idx for idx, _ in ranked] == [1, 2]


def test_retrieved_passages_are_rendered_in_source_order_with_one_based_labels():
    passages = _passages()
    later = max(range(len(passages)), key=lambda i: passage_key(passages[i]))
    earlier = min(range(len(passages)), key=lambda i: passage_key(passages[i]))
    text = format_retrieved_passages(passages, [(later, 3.0), (earlier, 2.0)])
    first_label = "Solution attempt 1, step 1:"
    assert text.startswith(first_label)
    assert "Solution attempt 2, step {}:".format(passages[later]["step"] + 1) in text


def test_format_query_follows_qwen3_embedding_card():
    assert format_query(" What is x? ", "Find it") == "Instruct: Find it\nQuery:What is x?"


def test_run_single_shot_rag_matches_ica_request_and_output_schema(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    record = {
        "benchmark": "aime24",
        "problem_id": "60",
        "question": "Find the residue modulo nine.",
        "instance": {"answer": "4"},
        "rollouts": ROLLOUTS,
    }
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    sent = []

    class FakeClient:
        def chat_completion(self, **kwargs):
            config = kwargs["config"]
            assert config.top_k is None
            assert config.thinking is True
            assert (config.temperature, config.top_p, config.max_tokens) == (1.0, 0.95, 128)
            assert kwargs["n"] == 1
            sent.append(kwargs["messages"])

            class Choice:
                text = "Combining the passages gives \\boxed{4}."
                finish_reason = "stop"

            class Parsed:
                choices = [Choice()]
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                raw_response = {}

            return Parsed()

    output_path = run_single_shot_rag(
        input_jsonl=input_path,
        output_dir=tmp_path / "out",
        retriever="bm25",
        top_m=2,
        model="qwen3.5-9b",
        client=FakeClient(),
        k=2,
        step_token_target=8,
        max_model_len=4096,
        max_tokens=128,
        prompt_token_buffer=32,
        token_counter=TransformersTokenCounter(FakeChatTokenizer()),
        passage_tokenizer=FakeWordTokenizer(),
        overwrite=True,
        max_workers=1,
    )

    assert output_path.name == "qwen3.5-9b_aime24_n2.jsonl"
    (messages,) = sent
    assert len(messages) == 1 and messages[0]["role"] == "user"
    prompt = messages[0]["content"]
    assert prompt.startswith("Given the following problem:\nFind the residue modulo nine.")
    assert "Solution attempt 1, step" in prompt

    out = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert out["aggregated_answer"] == "4"
    assert out["gold_answer"] == "4"
    assert out["retriever"] == "bm25" and out["top_m"] == 2
    assert 1 <= len(out["retrieved"]) <= 2
    assert out["token_usage"]["prompt_tokens"] == 10

    summary = evaluate.evaluate_file(output_path)
    assert summary["accuracy"] == 1.0


def test_dense_retrieval_checks_embedded_corpus(tmp_path):
    record = {"problem_id": "p1", "question": "Find the residue.", "rollouts": ROLLOUTS}
    corpus = collect_corpus([record], k=2, tokenizer=FakeWordTokenizer(), step_token_target=8, instruction="Find it")
    n = len(corpus["passage_texts"])
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n, 4)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    npz = tmp_path / "emb.npz"
    np.savez(
        npz,
        passage_problem_ids=np.asarray(corpus["passage_problem_ids"]),
        passage_solution_ids=np.asarray(corpus["passage_solution_ids"]),
        passage_steps=np.asarray(corpus["passage_steps"]),
        passage_digests=np.asarray(corpus["passage_digests"]),
        passage_embeddings=vectors.astype(np.float16),
        query_problem_ids=np.asarray(corpus["query_problem_ids"]),
        query_embeddings=vectors[:1].astype(np.float16),
    )
    entry = load_embeddings(npz)["p1"]
    passages = _passages()
    ranked = retrieve("qwen3emb", record["question"], passages, top_m=3, embedding_entry=entry)
    assert ranked[0][0] == 0 and len(ranked) == 3

    entry["digests"][0] = "tampered"
    with pytest.raises(ValueError):
        retrieve("qwen3emb", record["question"], passages, top_m=3, embedding_entry=entry)
