"""math_verify-backed grader for math rollouts."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Optional


@contextlib.contextmanager
def _maybe_suppress_grader_output():
    if os.environ.get("AGGMATH_SHOW_GRADER_OUTPUT") == "1":
        yield
        return

    quiet_loggers = [logging.getLogger("math_verify.parser"), logging.getLogger("math_verify.grader")]
    previous_levels = [logger.level for logger in quiet_loggers]
    for logger in quiet_loggers:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in zip(quiet_loggers, previous_levels):
            logger.setLevel(level)


def grade_prediction(gold: Optional[str], prediction_text: Optional[str]) -> bool:
    """Return True iff prediction_text is mathematically equivalent to gold.

    Uses HuggingFace math_verify (https://github.com/huggingface/Math-Verify).
    The gold answer is wrapped in \\boxed{...} so math_verify can extract it
    via its default LaTeX extraction. Any parse / verify failure (timeout,
    unparseable expression, etc.) is treated as a wrong answer rather than
    raising — eval should never crash on a single bad rollout.
    """
    if gold is None or prediction_text is None:
        return False
    if not str(gold).strip() or not str(prediction_text).strip():
        return False

    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ImportError(
            "math_verify is required for evaluation. Install it via "
            "`pip install math-verify`."
        ) from exc

    with _maybe_suppress_grader_output():
        try:
            gold_parsed = parse("\\boxed{{{}}}".format(gold), parsing_timeout=None)
        except Exception:
            return False
        if not gold_parsed:
            return False

        try:
            pred_parsed = parse(prediction_text, parsing_timeout=None)
        except Exception:
            return False
        if not pred_parsed:
            return False

        try:
            return bool(verify(gold_parsed, pred_parsed, timeout_seconds=None))
        except Exception:
            return False
