import json

from baselines.majority_voting.run_majority_voting import (
    cluster_votes,
    collect_votes,
    main,
    make_equivalence,
    pick_majority,
    vote_record,
)


def _rollouts(answers):
    return [{"rollout_idx": idx, "reasoning": "...", "answer": answer} for idx, answer in enumerate(answers)]


def test_collect_votes_skips_missing_answers_and_respects_k():
    votes = collect_votes(_rollouts(["1", None, " ", "2", "3"]), k=4)
    assert votes == [(0, "1"), (3, "2")]


def test_tie_goes_to_cluster_whose_first_vote_is_earliest():
    votes = [(0, "1"), (1, "2"), (2, "2"), (3, "1")]
    clusters = cluster_votes(votes, lambda a, b: a == b)
    winner = pick_majority(clusters)
    assert winner["answer"] == "1"
    assert winner["count"] == 2


def test_largest_cluster_wins():
    votes = [(0, "1"), (1, "2"), (2, "2"), (3, "3")]
    clusters = cluster_votes(votes, lambda a, b: a == b)
    assert pick_majority(clusters)["answer"] == "2"
    assert pick_majority([]) is None


def test_vote_record_merges_symbolically_equivalent_answers():
    record = {
        "benchmark": "aime24",
        "problem_id": "p1",
        "instance": {"answer": "\\frac{1}{2}"},
        "rollouts": _rollouts(["0.5", "1/3", "\\frac{1}{2}", "1/3", "\\dfrac{1}{2}"]),
    }
    result = vote_record(record, k=8, equivalent=make_equivalence({}), grade_cache={})
    assert result["majority_answer"] == "0.5"
    assert result["majority_count"] == 3
    assert result["correct"] is True
    assert result["n_votes"] == 5


def test_vote_record_without_votes_is_wrong():
    record = {"problem_id": "p2", "instance": {"answer": "7"}, "rollouts": _rollouts([None, None])}
    result = vote_record(record, k=8, equivalent=make_equivalence({}), grade_cache={})
    assert result["majority_answer"] is None
    assert result["correct"] is False


def test_main_writes_per_problem_and_summary_files(tmp_path):
    rollout_dir = tmp_path / "math_rollout" / "qwen3.5-2b"
    rollout_dir.mkdir(parents=True)
    records = [
        {"benchmark": "aime24", "problem_id": "a", "instance": {"answer": "204"},
         "rollouts": _rollouts(["204", "204", "17"])},
        {"benchmark": "aime24", "problem_id": "b", "instance": {"answer": "33"},
         "rollouts": _rollouts(["12", "12", "33"])},
    ]
    input_path = rollout_dir / "aime24_thinking_n8.jsonl"
    input_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    out_dir = tmp_path / "mv"
    assert main(["--inputs", str(rollout_dir), "--k", "8", "--output-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "qwen3.5-2b" / "aime24_thinking_n8_mv_k8.eval.json").read_text())
    assert summary["n_total"] == 2
    assert summary["n_correct"] == 1
    assert summary["accuracy"] == 0.5
    lines = (out_dir / "qwen3.5-2b" / "aime24_thinking_n8_mv_k8.jsonl").read_text().splitlines()
    assert [json.loads(line)["majority_answer"] for line in lines] == ["204", "12"]
