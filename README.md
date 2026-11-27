# Aggregation as Retrieval

Code for the CS646 (UMass Amherst, Fall 2026) project **Aggregation as
Retrieval: Aggregating Long Chain-of-Thought Math Solutions Beyond the Context
Window**, by Run-Ze Fan, Chanwoong Yoon, Wuao Liu, Shuyan Huang, and Hyungji Kim.

A solver samples eight long chain-of-thought solutions per problem, and the same
model then aggregates them. We propose **Agentic Aggregation**, which reads the
candidates through tools instead of placing them in its prompt, and compare it
with two other ways to read the candidates, all on the same saved inputs:

| Method | What the aggregator reads |
|---|---|
| In-context aggregation (ICA) | all eight traces in one prompt, truncated to fit the context |
| Single-shot RAG | the top-*m* steps of the candidates, retrieved once with the problem as the query (BM25 or Qwen3-Embedding-8B) |
| Agentic Aggregation (ours) | only what it asks for through tools: the answer list, per-candidate search, step reads, and summaries |

Agentic Aggregation splits each candidate into steps of at most 1,024 tokens and
gives the aggregator five tools: `get_answer` lists the candidates' answers,
`search_solution` searches one candidate with ROUGE-L, `get_reasoning_step`
reads up to five consecutive steps, `get_solution_summary` summarizes one
candidate in a separate call, and `finish` submits a solution with a boxed
answer.

Baselines are Pass@1, Pass@8 (the oracle for answer selection), and majority
voting. The models are Qwen3.5-2B, 4B, and 9B in thinking mode, and the
benchmarks are AIME 2024–2026 and HMMT February 2024–2025, 150 problems in all.

## Repository layout

| Path | Contents |
|---|---|
| `aggmath/rollout/` | candidate sampling against an OpenAI-compatible server |
| `aggmath/evaluation/` | Math-Verify grading and Pass@k |
| `aggmath/aggregation/` | Agentic Aggregation: runs the aggregator over the saved candidates |
| `baselines/prompted_aggregation/` | ICA with the AggLM prompt |
| `baselines/single_shot_rag/` | passage corpus, BM25 and embedding retrieval, single-shot RAG |
| `baselines/majority_voting/` | majority voting over Math-Verify-equivalent answers |
| `vendor/aggagent_core/` | the agent loop and tools of Agentic Aggregation, built on AggAgent's open-source code (Apache-2.0) |
| `scripts/` | experiment entry points (`run_*.sh`) and the vLLM launcher |
| `analysis/` | the tables and statistics in the report |
| `slurm/` | the job scripts we used on the UMass Unity cluster |
| `results/` | the result tables behind the report |
| `tests/` | unit tests |

## Setup

```bash
conda create -n aggmath python=3.11 -y
conda activate aggmath
pip install -U uv
uv pip install "vllm==0.19.0"   # brings torch and transformers
pip install -r requirements.txt
```

We used vLLM 0.19.0, torch 2.10.0, transformers 4.57.6, math-verify 0.9.0,
bm25s 0.3.8, and rouge-score 0.1.2, with these Hugging Face model revisions:

| Model | Revision |
|---|---|
| `Qwen/Qwen3.5-2B` | `15852e8` |
| `Qwen/Qwen3.5-4B` | `851bf6e` |
| `Qwen/Qwen3.5-9B` | `c202236` |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4c` |

The benchmarks load from Hugging Face: `math-ai/aime24`,
`MathArena/aime_2025`, `MathArena/aime_2026`, `MathArena/hmmt_feb_2024`, and
`MathArena/hmmt_feb_2025`.

## Reproducing the experiments

Run every command from the repository root. Each stage resumes: running it
again skips the problems already in its output file. We set
`REQUEST_TIMEOUT=10800` for every client, because shorter timeouts silently
retried long generations and kept only the shorter ones.

**1. Serve a model** on one GPU. Aggregation needs `MAX_MODEL_LEN=262144`.

```bash
MODEL=Qwen/Qwen3.5-9B DATA_PARALLEL_SIZE=1 MAX_MODEL_LEN=65536 PORT=8000 \
  bash scripts/serve_qwen35.sh
```

**2. Sample eight candidates per problem** (temperature 1.0, top-p 0.95,
top-k 20, at most 32,768 new tokens).

```bash
MODEL=qwen3.5-9b BASE_URL=http://127.0.0.1:8000/v1 REQUEST_TIMEOUT=10800 \
  bash scripts/run_rollout.sh
# writes outputs/math_rollout/qwen3.5-9b/<benchmark>_thinking_n8.jsonl
```

**3. Pass@1 and Pass@8.** `slurm/grader_timeout.py` caps each Math-Verify call
at 60 seconds.

```bash
python slurm/grader_timeout.py aggmath.evaluation.run_eval \
  --inputs outputs/math_rollout/qwen3.5-9b/aime24_thinking_n8.jsonl \
  --k 1 8 --output-dir outputs/eval --write-per-problem
```

**4. Majority voting.**

```bash
python slurm/grader_timeout.py baselines.majority_voting.run_majority_voting \
  --inputs outputs/math_rollout/qwen3.5-9b --k 8 --output-dir outputs/majority_voting
```

**5. ICA.**

```bash
MODEL=Qwen/Qwen3.5-9B REQUEST_TIMEOUT=10800 bash scripts/run_prompted_aggregation_9b.sh
```

Use `scripts/run_prompted_aggregation_4b.sh` for 4B. For 2B, run the 9B script
with `MODEL`, `INPUT_DIR`, `OUTPUT_DIR`, and `LOG_DIR` set for 2B, as
`slurm/prompted.sbatch` does.

**6. Single-shot RAG** with BM25 at the default depth, then the depth sweep.

```bash
MODEL=Qwen/Qwen3.5-9B MODEL_SLUG=qwen3.5-9b RETRIEVER=bm25 TOP_M=16 REQUEST_TIMEOUT=10800 \
  bash scripts/run_single_shot_rag.sh
for m in 4 8 32 64 128; do
  MODEL=Qwen/Qwen3.5-9B MODEL_SLUG=qwen3.5-9b RETRIEVER=bm25 TOP_M=$m REQUEST_TIMEOUT=10800 \
    bash scripts/run_single_shot_rag.sh
done
```

For Qwen3-Embedding-8B, build and embed the passage corpus first (the `embed`
stage needs a GPU), then run with `RETRIEVER=qwen3emb`.

```bash
python -m baselines.single_shot_rag.embed_passages --stage corpus --inputs outputs/math_rollout/qwen3.5-9b
python -m baselines.single_shot_rag.embed_passages --stage embed --inputs outputs/math_rollout/qwen3.5-9b
```

**7. Agentic Aggregation.** The 2B and 4B scripts work the same
way.

```bash
BASE_URL=http://127.0.0.1:8000/v1 REQUEST_TIMEOUT=10800 bash scripts/run_agentic_aggregation_9b.sh
```

**8. Tables and statistics.**

```bash
python analysis/collect_accuracy.py                          # accuracy by method and benchmark
python slurm/grader_timeout.py analysis.retrieval_quality    # precision of the passages read
python analysis/aggregation_cost.py                          # aggregation tokens per problem
python analysis/candidate_categories.py                      # problems solved by candidate category
python analysis/final_report_stats.py                        # paired tests, agent tool use, cut-off candidates
```

The first three write to `outputs/analysis/`. `results/` holds the outputs
behind the report:

| Report item | Source |
|---|---|
| Table 1, accuracy | `results/accuracy.md` |
| Figure 1, accuracy against tokens | `results/accuracy.md`, `results/cost.md` |
| Table 2, candidate categories | `results/candidate_categories.txt` |
| Paired tests, tool use, cut-off candidates | `results/final_report_stats.txt` |
| Precision of the passages read | `results/retrieval_quality.md` |

## Running on Slurm

`slurm/` holds the job scripts we used on the UMass Unity cluster, described in
`slurm/README.md`. Each GPU job starts vLLM on one A100-80GB, runs one stage,
and stops the server; for example, `bash slurm/submit.sh rollout` submits all
fifteen rollout jobs. The account, partitions, and paths in `slurm/*.sbatch` and
`slurm/env.sh` belong to that cluster, so change them before running elsewhere.

## Evaluation conventions

- Every accuracy divides by all 30 problems of a benchmark, and a missing
  answer counts as wrong.
- A candidate cut off before its final answer is graded on the last boxed
  expression in its reasoning trace.
- Each method runs once per problem on the same saved candidates.

## Acknowledgments and license

- The agent loop in `vendor/aggagent_core/` builds on the open-source code of
  [AggAgent](https://github.com/princeton-pli/AggAgent) (commit `66b697d`,
  Apache-2.0). `vendor/aggagent_core/UPSTREAM.md` lists our changes.
- ICA uses the prompt template of AggLM (Zhao et al., 2025).
- The rest of the code is released under the MIT License (`LICENSE`).
