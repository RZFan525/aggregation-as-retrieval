# Vendored From AggAgent

This directory vendors a small subset of the aggregation core from:

- Repository: `https://github.com/princeton-pli/AggAgent`
- Upstream commit: `66b697deb01b03522544a9e6a0560b51a970f36a`
- Upstream license: Apache-2.0

Files copied here are intentionally limited to the aggregation core needed for
AggMath's math-focused experiments. They are kept separate from project-native
code so that:

1. math-specific changes are easy to find,
2. upstream-derived code remains attributable,
3. future rebases or diff checks against upstream stay manageable.

Current vendored scope:

- `agent.py`
- `tools.py`
- `prompts.py`
- `strategies/base.py`
- `strategies/pass_strategy.py`
- `strategies/mv.py`
- `strategies/bon.py`
- `strategies/llm_based.py`

Guideline:

- Put new math logic in `aggmath/` whenever possible.
- Only modify files here when adapting upstream aggregation behavior itself.

## Changes in this repository

To aggregate math solutions, we changed `agent.py`, `prompts.py`, and
`tools.py`; each of these files starts with a notice saying so.

- `agent.py` takes math solutions (a reasoning trace and an answer each),
  counts tokens with the Qwen3.5 tokenizer, and configures the summary call and
  the search backend.
- `prompts.py` asks the agent to verify math reasoning step by step.
- `tools.py` splits each reasoning trace into steps of a fixed token budget,
  extracts boxed answers, and asks for a boxed answer in `finish`.

The other vendored files are unchanged.
