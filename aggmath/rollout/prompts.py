"""Prompt helpers for benchmark rollout."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_math_messages(
    question: str,
    benchmark: Optional[str] = None,
    instance: Optional[Dict[str, Any]] = None,
    extra_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    del benchmark, instance
    user_content = "{}\n\n{}".format(MATH_INSTRUCTION, question.strip())
    if extra_instruction:
        user_content = "{}\n\n{}".format(user_content, extra_instruction.strip())
    return [{"role": "user", "content": user_content}]
