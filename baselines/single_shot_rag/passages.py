"""Per-problem passage corpus D_x and retrieval for single-shot RAG.

Passages are exactly the reasoning steps the agentic aggregator works with:
each of the first k rollouts is split by
vendor.aggagent_core.tools.build_solution_environment (about step_token_target
tokens per step, using the same Qwen3.5 tokenizer), and the BM25 retriever is
the vendored global step index used by AggAgent's search_solution tool.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

from vendor.aggagent_core.tools import (
    _BM25GlobalStepIndex,
    _build_bm25_documents,
    build_solution_environment,
)

DEFAULT_QUERY_INSTRUCTION = (
    "Given a competition math problem, retrieve passages from candidate solutions "
    "that help solve the problem or verify its answer"
)


def build_passages(
    rollouts: Sequence[Any],
    k: int,
    tokenizer: Any,
    step_token_target: int = 1024,
) -> List[Dict[str, Any]]:
    """Return D_x as dicts with solution_id, step, tokens, source and content."""
    selected = [rollout for rollout in list(rollouts)[:k] if isinstance(rollout, dict)]
    if len(selected) < k:
        raise ValueError("Record has only {} usable rollouts, fewer than k={}.".format(len(selected), k))
    solutions = build_solution_environment(
        selected,
        tokenizer=tokenizer,
        step_token_target=step_token_target,
    )
    return _build_bm25_documents(solutions)


def passage_key(passage: Dict[str, Any]) -> Tuple[int, int]:
    return int(passage["solution_id"]), int(passage["step"])


def content_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def bm25_rank(query: str, passages: Sequence[Dict[str, Any]], top_m: int) -> List[Tuple[int, float]]:
    """Top-m passage indices by BM25 (only passages with a positive score)."""
    if not passages:
        return []
    positions = {passage_key(passage): idx for idx, passage in enumerate(passages)}
    index = _BM25GlobalStepIndex(list(passages))
    return [(positions[passage_key(doc)], score) for doc, score in index.search(query, k=top_m)]


def dense_rank(query_vector: Any, passage_vectors: Any, top_m: int) -> List[Tuple[int, float]]:
    """Top-m passage indices by cosine similarity of L2-normalized vectors."""
    import numpy as np

    if len(passage_vectors) == 0:
        return []
    scores = np.asarray(passage_vectors, dtype=np.float32) @ np.asarray(query_vector, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:top_m]
    return [(int(idx), float(scores[idx])) for idx in order]


def format_query(question: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """Qwen3-Embedding query format from its model card."""
    return "Instruct: {}\nQuery:{}".format(instruction, question.strip())


def format_retrieved_passages(
    passages: Sequence[Dict[str, Any]],
    ranked: Sequence[Tuple[int, float]],
) -> str:
    """Render retrieved passages in source order for the ICA prompt's {solutions} slot."""
    chosen = sorted((passages[idx] for idx, _ in ranked), key=passage_key)
    chunks = [
        "Solution attempt {}, step {}:\n{}".format(
            int(passage["solution_id"]) + 1,
            int(passage["step"]) + 1,
            str(passage["content"]).strip(),
        )
        for passage in chosen
    ]
    return "\n\n".join(chunks)
