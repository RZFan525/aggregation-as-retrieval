"""Embed the single-shot RAG passage corpora with Qwen3-Embedding.

The work is split into two stages so the GPU never waits on CPU work.

- corpus (CPU job): rebuild D_x for each rollout JSONL exactly as
  run_single_shot_rag.py does (same rollouts, Qwen3.5 tokenizer and step size).
  Tokenize every passage and every instruction-formatted query with the
  embedding tokenizer, truncate to --max-tokens, and write
  <output-dir>/<model dir>/<input stem>.corpus.npz. Truncation keeps the
  trailing <|endoftext|>, the token Qwen3-Embedding pools over.
- embed (GPU job): embed every corpus file with offline vLLM and write
  <output-dir>/<model dir>/<input stem>.npz. Vectors are L2-normalized
  float16. Passage keys and content digests are stored too, so the RAG runner
  can check that it rebuilt the same corpus.

Any error exits the process at once, so a failed job releases its GPU instead
of hanging on vLLM's engine process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aggmath.evaluation.run_eval import _expand_input_paths, _read_jsonl
from aggmath.rollout.qwen_api import normalize_model_name
from baselines.single_shot_rag.passages import (
    DEFAULT_QUERY_INSTRUCTION,
    build_passages,
    content_digest,
    format_query,
)
from baselines.single_shot_rag.run_single_shot_rag import load_passage_tokenizer

_TOKENIZERS: Dict[str, Any] = {}


def collect_corpus(
    records: Sequence[Dict[str, Any]],
    k: int,
    tokenizer: Any,
    step_token_target: int,
    instruction: str,
) -> Dict[str, List[Any]]:
    corpus: Dict[str, List[Any]] = {
        "passage_problem_ids": [],
        "passage_solution_ids": [],
        "passage_steps": [],
        "passage_digests": [],
        "passage_texts": [],
        "query_problem_ids": [],
        "query_texts": [],
    }
    for record in records:
        problem_id = str(record.get("problem_id"))
        passages = build_passages(record.get("rollouts") or [], k, tokenizer, step_token_target)
        for passage in passages:
            text = str(passage["content"])
            corpus["passage_problem_ids"].append(problem_id)
            corpus["passage_solution_ids"].append(int(passage["solution_id"]))
            corpus["passage_steps"].append(int(passage["step"]))
            corpus["passage_digests"].append(content_digest(text))
            corpus["passage_texts"].append(text)
        corpus["query_problem_ids"].append(problem_id)
        corpus["query_texts"].append(format_query(str(record.get("question") or ""), instruction))
    return corpus


def tokenize_truncated(texts: Sequence[str], tokenizer: Any, max_tokens: int) -> Tuple[List[List[int]], int]:
    """Tokenize with special tokens; cut to max_tokens but keep the trailing special tokens."""
    suffix = list(tokenizer("")["input_ids"])
    token_lists: List[List[int]] = []
    n_truncated = 0
    for text in texts:
        ids = list(tokenizer(text)["input_ids"])
        if len(ids) > max_tokens:
            ids = ids[: max_tokens - len(suffix)] + suffix
            n_truncated += 1
        token_lists.append(ids)
    return token_lists, n_truncated


def _pack(token_lists: Sequence[Sequence[int]]) -> Tuple[Any, Any]:
    import numpy as np

    offsets = np.zeros(len(token_lists) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(ids) for ids in token_lists])
    flat = np.fromiter((tok for ids in token_lists for tok in ids), dtype=np.int32, count=int(offsets[-1]))
    return flat, offsets


def _unpack(flat: Any, offsets: Any) -> List[List[int]]:
    return [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def _tokenizer(name: str, loader: Any) -> Any:
    if name not in _TOKENIZERS:
        _TOKENIZERS[name] = loader(name)
    return _TOKENIZERS[name]


def build_corpus_file(
    path: Path,
    output_path: Path,
    embedding_model: str,
    k: int,
    step_token_target: int,
    instruction: str,
    max_tokens: int,
) -> str:
    import numpy as np
    from transformers import AutoTokenizer

    model = normalize_model_name(path.parent.name)
    passage_tokenizer = _tokenizer(model, load_passage_tokenizer)
    embedding_tokenizer = _tokenizer(embedding_model, AutoTokenizer.from_pretrained)
    corpus = collect_corpus(_read_jsonl(path), k, passage_tokenizer, step_token_target, instruction)
    passage_tokens, n_cut = tokenize_truncated(corpus["passage_texts"], embedding_tokenizer, max_tokens)
    query_tokens, _ = tokenize_truncated(corpus["query_texts"], embedding_tokenizer, max_tokens)
    passage_flat, passage_offsets = _pack(passage_tokens)
    query_flat, query_offsets = _pack(query_tokens)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        passage_problem_ids=np.asarray(corpus["passage_problem_ids"]),
        passage_solution_ids=np.asarray(corpus["passage_solution_ids"], dtype=np.int32),
        passage_steps=np.asarray(corpus["passage_steps"], dtype=np.int32),
        passage_digests=np.asarray(corpus["passage_digests"]),
        passage_token_flat=passage_flat,
        passage_token_offsets=passage_offsets,
        query_problem_ids=np.asarray(corpus["query_problem_ids"]),
        query_token_flat=query_flat,
        query_token_offsets=query_offsets,
    )
    return "{}: {} passages ({} truncated to {} tokens), {} queries".format(
        output_path, len(passage_tokens), n_cut, max_tokens, len(query_tokens)
    )


def embed_corpus_file(llm: Any, corpus_path: Path, output_path: Path, metadata: Dict[str, Any]) -> str:
    import numpy as np
    from vllm.inputs import TokensPrompt

    data = np.load(corpus_path, allow_pickle=False)

    def embed(flat: Any, offsets: Any) -> Any:
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in _unpack(flat, offsets)]
        vectors = np.asarray([out.outputs.embedding for out in llm.embed(prompts)], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-12, None)).astype(np.float16)

    passage_embeddings = embed(data["passage_token_flat"], data["passage_token_offsets"])
    query_embeddings = embed(data["query_token_flat"], data["query_token_offsets"])
    np.savez(
        output_path,
        passage_problem_ids=data["passage_problem_ids"],
        passage_solution_ids=data["passage_solution_ids"],
        passage_steps=data["passage_steps"],
        passage_digests=data["passage_digests"],
        passage_embeddings=passage_embeddings,
        query_problem_ids=data["query_problem_ids"],
        query_embeddings=query_embeddings,
    )
    metadata = dict(metadata, n_passages=len(passage_embeddings), n_queries=len(query_embeddings))
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return "{}: {} passages, {} queries".format(output_path, len(passage_embeddings), len(query_embeddings))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed single-shot RAG passages with Qwen3-Embedding.")
    parser.add_argument("--stage", required=True, choices=["corpus", "embed"])
    parser.add_argument("--inputs", nargs="+", required=True, help="Rollout JSONL files, directories or globs.")
    parser.add_argument("--output-dir", default="outputs/SingleShot_RAG/embeddings")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--step-token-target", type=int, default=1024)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--max-tokens", type=int, default=32768, help="Truncate embedding inputs to this many tokens.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [path for path in _expand_input_paths(args.inputs) if not path.name.endswith(".eval.jsonl")]
    output_root = Path(args.output_dir).expanduser().resolve()

    def target(path: Path, suffix: str) -> Path:
        return output_root / path.parent.name / "{}{}".format(path.stem, suffix)

    if args.stage == "corpus":
        todo = [p for p in inputs if args.overwrite or not target(p, ".corpus.npz").exists()]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [
                pool.submit(build_corpus_file, p, target(p, ".corpus.npz"), args.embedding_model, args.k,
                            args.step_token_target, args.query_instruction, args.max_tokens)
                for p in todo
            ]
            for future in futures:
                print(future.result(), flush=True)
        return 0

    from vllm import LLM

    missing = [str(target(p, ".corpus.npz")) for p in inputs if not target(p, ".corpus.npz").exists()]
    if missing:
        raise FileNotFoundError("Run --stage corpus first; missing: {}".format(missing))
    todo = [p for p in inputs if args.overwrite or not target(p, ".npz").exists()]
    if not todo:
        print("all embeddings exist")
        return 0
    llm = LLM(
        model=args.embedding_model,
        runner="pooling",
        convert="embed",
        max_model_len=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    metadata = {
        "embedding_model": args.embedding_model,
        "k": args.k,
        "step_token_target": args.step_token_target,
        "query_instruction": args.query_instruction,
        "max_tokens": args.max_tokens,
    }
    for path in todo:
        print(embed_corpus_file(llm, target(path, ".corpus.npz"), target(path, ".npz"), dict(metadata, input=str(path))),
              flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # vLLM's engine process can keep the interpreter alive after an error; exit hard.
    os._exit(exit_code)
