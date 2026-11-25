"""Helpers for extracting boxed math answers from model text."""

from __future__ import annotations

from typing import Optional, Tuple


def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    last_match: Optional[str] = None
    cursor = 0
    needle = "\\boxed{"
    while True:
        start = text.find(needle, cursor)
        if start == -1:
            break
        brace_start = start + len(needle) - 1
        boxed, end = _read_braced(text, brace_start)
        if boxed is not None:
            last_match = boxed
            cursor = end
        else:
            cursor = start + len(needle)
    return last_match.strip() if last_match else None


def _read_braced(text: str, brace_start: int) -> Tuple[Optional[str], int]:
    if brace_start >= len(text) or text[brace_start] != "{":
        return None, brace_start
    depth = 0
    start = brace_start + 1
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx], idx + 1
    return None, len(text)
