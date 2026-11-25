#!/usr/bin/env python3
"""Run a module with a 60-second limit on math_verify parse/verify calls.

Since commit 9a04546 (2026-05-31), aggmath/evaluation/grader.py calls
math_verify with parsing_timeout=None and timeout_seconds=None, so grading can
run forever on a pathological prediction. run_eval grades the full text of any
rollout without a \\boxed{} answer, and one such text (qwen3.5-4b aime26_013,
rollout 4) never finished.

The original April 2026 evaluation used math_verify's default 5s limits. As
agreed on 2026-10-25, we use 60s instead. When a parse or verify runs over the
limit, math_verify gives up and grade_prediction counts the answer as wrong,
which is how a timeout behaved in April. The limits are signal based, so they
only work in the main thread; that is where the evaluation entry points grade.

Usage: python slurm/grader_timeout.py <module> [module args...]
  e.g. python slurm/grader_timeout.py aggmath.evaluation.run_eval --inputs f.jsonl --k 1 8
"""

import runpy
import sys
from pathlib import Path

import math_verify

# Run as a script, sys.path[0] is slurm/; the repo modules live one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIMEOUT_SECONDS = 60
_parse, _verify = math_verify.parse, math_verify.verify


def parse(*args, **kwargs):
    if kwargs.get("parsing_timeout") is None:
        kwargs["parsing_timeout"] = TIMEOUT_SECONDS
    return _parse(*args, **kwargs)


def verify(*args, **kwargs):
    if kwargs.get("timeout_seconds") is None:
        kwargs["timeout_seconds"] = TIMEOUT_SECONDS
    return _verify(*args, **kwargs)


# grade_prediction imports parse/verify from math_verify on every call, so it picks these up.
math_verify.parse = parse
math_verify.verify = verify

if __name__ == "__main__":
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
