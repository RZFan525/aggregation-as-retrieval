# Modified from AggAgent (https://github.com/princeton-pli/AggAgent, commit 66b697d) to aggregate math solutions; see UPSTREAM.md.
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional, Union

try:
    import bm25s as _bm25s
except ImportError:  # pragma: no cover - exercised when optional dep is missing.
    _bm25s = None

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except ImportError:  # pragma: no cover - exercised in minimal test environments.
    _rouge_scorer = None

class _SimpleRougeScore:
    def __init__(self, recall: float):
        self.recall = recall


class _SimpleRougeScorer:
    def score(self, query: str, text: str) -> dict[str, _SimpleRougeScore]:
        query_terms = set(re.findall(r"\S+", query.lower()))
        text_terms = set(re.findall(r"\S+", text.lower()))
        recall = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
        return {"rougeL": _SimpleRougeScore(recall)}


_scorer = (
    _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    if _rouge_scorer is not None
    else _SimpleRougeScorer()
)


class BaseTool:
    """Minimal local stand-in for qwen_agent.tools.base.BaseTool."""

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}


def register_tool(_name: str, allow_overwrite: bool = False):
    """No-op decorator that preserves AggAgent's class definitions."""

    def decorator(cls):
        return cls

    return decorator


def _get_content(message: dict, key: str = "content") -> str:
    """Extract text content from a message dict (handles string or list formats)."""
    value = message.get(key, "")
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        if isinstance(value[0], dict):
            text = value[0].get("text") or ""
            if key == "content":
                recipient = message.get("recipient")
                name = message.get("name")
                if name:
                    text = f"[Tool Response: {name}]\n{text}"
                elif recipient:
                    text = f"[Tool Call: {recipient}]\n{text}"
            return text
    return ""


def _rouge_l_recall(query: str, text: str) -> float:
    """ROUGE-L recall of query against text."""
    if not query or not text:
        return 0.0
    return _scorer.score(query, text)['rougeL'].recall


def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to first n words."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == max_words:
            return text[:m.end()] + '\n[... truncated]'
    return text


def _count_tokens_approx(messages: list, chars_per_token: float = 4.0) -> int:
    """Approximate token count for a list of messages."""
    total_chars = 0
    for msg in messages:
        for key in ["role", "reasoning_content", "reasoning", "content"]:
            value = msg.get(key)
            if isinstance(value, str):
                total_chars += len(value)
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False))
    return int(total_chars / chars_per_token)


def count_text_tokens(text: str, tokenizer: Any) -> int:
    """Count text tokens with the provided Qwen tokenizer."""
    if not text:
        return 0
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    input_ids = _extract_input_ids(token_ids)
    return len(input_ids)


def _extract_input_ids(tokenized: Any) -> Any:
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return tokenized["input_ids"]
    input_ids = getattr(tokenized, "input_ids", None)
    if input_ids is not None:
        return input_ids
    return tokenized


def _extract_boxed_answer(text: Optional[str]) -> Optional[str]:
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


def _read_braced(text: str, brace_start: int) -> tuple[Optional[str], int]:
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


def _extract_reasoning(run: dict) -> str:
    reasoning = run.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    messages = run.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for key in ("reasoning", "reasoning_content", "content"):
                content = _get_content(message, key)
                if content:
                    parts.append(content)
        if parts:
            return "\n\n".join(parts).strip()

    prediction = run.get("prediction")
    if isinstance(prediction, str) and prediction.strip():
        return prediction.strip()
    content = run.get("content")
    return str(content).strip() if content is not None else ""


def _extract_answer(run: dict) -> Optional[str]:
    answer = run.get("answer")
    if answer is None:
        return None
    text = str(answer).strip()
    return text or None


def _iter_solution_runs(run_results: Any) -> list[dict[str, Any]]:
    """Return candidate rollout dicts from an AggMath record or rollout list."""
    if isinstance(run_results, dict):
        rollouts = run_results.get("rollouts")
        if isinstance(rollouts, list):
            return [rollout for rollout in rollouts if isinstance(rollout, dict)]
        return [run_results]

    if isinstance(run_results, list):
        runs: list[dict[str, Any]] = []
        for item in run_results:
            if not isinstance(item, dict):
                continue
            rollouts = item.get("rollouts")
            if isinstance(rollouts, list):
                runs.extend(rollout for rollout in rollouts if isinstance(rollout, dict))
            else:
                runs.append(item)
        return runs

    return []


def _split_reasoning_atoms(text: str, tokenizer: Any, step_token_target: int) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
    atoms: list[str] = []
    sentence_pattern = re.compile(r"(?<=[\.\?\!。？！；;])\s+")
    for block in blocks:
        if count_text_tokens(block, tokenizer) <= step_token_target:
            atoms.append(block)
            continue
        sentences = [part.strip() for part in sentence_pattern.split(block) if part.strip()]
        if len(sentences) <= 1:
            atoms.append(block)
        else:
            atoms.extend(sentences)
    return atoms


def _pack_reasoning_steps(
    atoms: list[str],
    tokenizer: Any,
    step_token_target: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        content = "\n".join(current).strip()
        steps.append(
            {
                "step": len(steps),
                "content": content,
                "tokens": count_text_tokens(content, tokenizer),
            }
        )
        current.clear()

    for atom in atoms:
        if not current:
            current.append(atom)
            if count_text_tokens(atom, tokenizer) > step_token_target:
                flush()
            continue

        candidate = "\n".join(current + [atom])
        if count_text_tokens(candidate, tokenizer) > step_token_target:
            flush()
        current.append(atom)
        if count_text_tokens(atom, tokenizer) > step_token_target:
            flush()

    flush()
    return steps


def build_solution_environment(
    run_results: Any,
    tokenizer: Any,
    step_token_target: int = 1024,
) -> list[dict[str, Any]]:
    """Convert candidate math outputs into hidden solution records."""
    solutions: list[dict[str, Any]] = []
    for solution_id, run in enumerate(_iter_solution_runs(run_results)):
        reasoning = _extract_reasoning(run)
        answer = _extract_answer(run)
        atoms = _split_reasoning_atoms(reasoning, tokenizer, step_token_target)
        steps = _pack_reasoning_steps(atoms, tokenizer, step_token_target)
        solutions.append(
            {
                "solution_id": solution_id,
                "answer": answer,
                "reasoning": reasoning,
                "steps": steps,
                "tokens": count_text_tokens(reasoning, tokenizer),
            }
        )
    return solutions


def format_metadata(solutions: list) -> str:
    """Format solution metadata for inclusion in the user prompt."""
    blocks = []
    for solution in solutions:
        solution_id = solution.get("solution_id")
        answer = solution.get("answer")
        answer_text = str(answer).strip() if answer is not None else "None"
        if not answer_text:
            answer_text = "None"
        blocks.append(
            "Solution {}: {} steps, {} tokens, answer: {}".format(
                solution_id,
                len(solution.get("steps", [])),
                int(solution.get("tokens", 0) or 0),
                answer_text,
            )
        )

    return "\n\n".join(blocks)


def _validate_solution_id(solution_id: Any, solutions: list, tool_name: str) -> Union[int, str]:
    if not isinstance(solution_id, int):
        return f"[{tool_name}] 'solution_id' must be an integer"
    n = len(solutions)
    if solution_id < 0 or solution_id >= n:
        return f"[{tool_name}] 'solution_id' must be 0-{n - 1}"
    return solution_id


@register_tool("get_answer", allow_overwrite=True)
class GetAnswerTool(BaseTool):
    name = "get_answer"
    description = "Retrieves candidate extracted answers and size metadata. Returns {solution_id, answer, steps, tokens} entries."
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> list:
        solutions = kwargs.get("solutions", [])
        results = []
        for solution in solutions:
            results.append(
                {
                    "solution_id": solution.get("solution_id"),
                    "answer": solution.get("answer"),
                    "steps": len(solution.get("steps", [])),
                    "tokens": int(solution.get("tokens", 0) or 0),
                }
            )
        return results

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


# Backward-compatible Python symbol for older imports. The exposed tool name is
# still get_answer.
GetAnswersTool = GetAnswerTool


@register_tool("get_reasoning_step", allow_overwrite=True)
class GetReasoningStepTool(BaseTool):
    name = "get_reasoning_step"
    description = "Reads the full content of a contiguous range of reasoning steps from a solution (max 5). Use after search_solution to inspect a step with surrounding context."
    parameters = {
        "type": "object",
        "properties": {
            "solution_id": {"type": "integer", "description": "0-based solution index."},
            "start_step": {"type": "integer", "description": "0-based start reasoning step (inclusive)."},
            "end_step": {"type": "integer", "description": "0-based end reasoning step (inclusive); end_step - start_step <= 4."}
        },
        "required": ["solution_id", "start_step", "end_step"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            solution_id = params["solution_id"]
            start_step = params["start_step"]
            end_step = params["end_step"]
        except Exception:
            return "[get_reasoning_step] Invalid request format: Input must be a JSON object containing 'solution_id', 'start_step', and 'end_step' fields"

        solutions = kwargs.get("solutions", [])
        validated = _validate_solution_id(solution_id, solutions, self.name)
        if isinstance(validated, str):
            return validated
        steps = solutions[validated].get("steps", [])
        n = len(steps)
        if n == 0:
            return []
        start_step = max(0, min(start_step, n - 1))
        end_step = max(0, min(end_step, n - 1))
        if start_step > end_step:
            start_step = end_step
        if end_step - start_step > 4:
            end_step = start_step + 4
        start_0 = start_step
        end_0 = end_step

        result = []
        for step_idx in range(start_0, end_0 + 1):
            step = steps[step_idx]
            entry = {
                "solution_id": solution_id,
                "step": int(step.get("step", step_idx)),
                "tokens": int(step.get("tokens", 0) or 0),
            }
            content = step.get("content", "")
            if content:
                entry["content"] = truncate_text(content, 600)
            result.append(entry)
        return result

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


@register_tool("search_solution", allow_overwrite=True)
class SearchSolutionTool(BaseTool):
    name = "search_solution"
    description = "Searches for keywords or phrases within a single solution's reasoning steps. Returns top matching steps ranked by relevance score."
    parameters = {
        "type": "object",
        "properties": {
            "solution_id": {"type": "integer", "description": "0-based solution index to search within."},
            "query": {"type": "string", "description": "Search term or phrase."},
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10).", "default": 5}
        },
        "required": ["solution_id", "query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _score_solution(self, solution_id: int, solution: dict, query: str):
        scored = []
        for step_idx, step in enumerate(solution.get("steps", [])):
            content = step.get("content", "") or ""
            score = _rouge_l_recall(query, content)
            if score > 0:
                scored.append((score, solution_id, step_idx, step))
        return scored

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[search_solution] Invalid request format: Input must be a JSON object containing 'query' field"

        solution_id = params.get("solution_id")
        if solution_id is None:
            return "[search_solution] 'solution_id' is required"
        max_results = min(params.get("k", 5), 10)
        solutions = kwargs.get("solutions", [])

        validated = _validate_solution_id(solution_id, solutions, self.name)
        if isinstance(validated, str):
            return validated
        scored = self._score_solution(validated, solutions[validated], query)
        scored.sort(key=lambda x: -x[0])

        matches = []
        for score, matched_solution_id, step_idx, step in scored[:max_results]:
            content = step.get("content", "") or ""
            match_entry = {
                "solution_id": matched_solution_id,
                "step": int(step.get("step", step_idx)),
                "tokens": int(step.get("tokens", 0) or 0),
                "score": round(score, 3),
            }
            if content:
                match_entry["content"] = truncate_text(content)
            matches.append(match_entry)

        if not matches:
            return f"[search_solution] No matches found for '{query}' in solution {solution_id}"
        return matches

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


class _BM25GlobalStepIndex:
    def __init__(self, documents: list[dict[str, Any]]):
        if _bm25s is None:
            raise ImportError(
                "bm25s is required for BM25 search_solution. Install it via "
                "`pip install bm25s` or select --search-tool rouge."
            )
        self.documents = documents
        corpus = [doc["content"] for doc in documents]
        self.retriever = _bm25s.BM25(corpus=documents)
        self.retriever.index(_bm25s.tokenize(corpus))

    def search(self, query: str, k: int) -> list[tuple[dict[str, Any], float]]:
        if not query.strip() or not self.documents:
            return []

        query_tokens = _bm25s.tokenize(query)
        docs, scores = self.retriever.retrieve(
            query_tokens,
            k=min(k, len(self.documents)),
        )
        doc_row = _first_result_row(docs)
        score_row = _first_result_row(scores)

        results: list[tuple[dict[str, Any], float]] = []
        for doc, score in zip(doc_row, score_row):
            numeric_score = float(score)
            if numeric_score <= 0:
                continue
            if isinstance(doc, dict):
                results.append((doc, numeric_score))
        return results


def _first_result_row(value: Any) -> list:
    try:
        return list(value[0])
    except Exception:
        return []


def _build_bm25_documents(solutions: list) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for fallback_solution_id, solution in enumerate(solutions):
        solution_id = solution.get("solution_id", fallback_solution_id)
        for fallback_step_idx, step in enumerate(solution.get("steps", [])):
            content = str(step.get("content", "") or "").strip()
            if not content:
                continue
            step_idx = int(step.get("step", fallback_step_idx))
            source = "Solution {}, step {}".format(solution_id, step_idx)
            documents.append(
                {
                    "solution_id": solution_id,
                    "step": step_idx,
                    "tokens": int(step.get("tokens", 0) or 0),
                    "source": source,
                    "content": content,
                }
            )
    return documents


def _get_bm25_global_index(
    solutions: list,
    tool_state: Optional[dict[str, Any]] = None,
) -> _BM25GlobalStepIndex:
    cache_key = "search_solution_bm25_global_index"
    if isinstance(tool_state, dict) and cache_key in tool_state:
        return tool_state[cache_key]

    index = _BM25GlobalStepIndex(_build_bm25_documents(solutions))
    if isinstance(tool_state, dict):
        tool_state[cache_key] = index
    return index


@register_tool("search_solution", allow_overwrite=True)
class SearchAllSolutionsBM25Tool(BaseTool):
    name = "search_solution"
    description = (
        "Searches all candidate solution reasoning steps with BM25 lexical retrieval. "
        "Use concise keywords, variables, formulas, numbers, or theorem names. "
        "Returns top matching steps labeled by source, e.g. Solution 3, step 5."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Short BM25 query with exact mathematical terms, symbols, variables, formulas, or numbers.",
            },
            "k": {"type": "integer", "description": "Max matches to return (default 5, max 10).", "default": 5}
        },
        "required": ["query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[search_solution] Invalid request format: Input must be a JSON object containing 'query' field"

        if not isinstance(query, str) or not query.strip():
            return "[search_solution] 'query' must be a non-empty string"

        max_results = min(max(int(params.get("k", 5)), 1), 10)
        solutions = kwargs.get("solutions", [])
        tool_state = kwargs.get("tool_state")
        index = _get_bm25_global_index(solutions, tool_state=tool_state)
        results = index.search(query, max_results)

        matches = []
        for doc, score in results:
            source = str(doc["source"])
            content = doc.get("content", "")
            match_entry = {
                "source": source,
                "solution_id": doc.get("solution_id"),
                "step": int(doc.get("step", 0)),
                "tokens": int(doc.get("tokens", 0) or 0),
                "score": round(float(score), 3),
            }
            if content:
                match_entry["content"] = "[{}]\n{}".format(source, truncate_text(content))
            matches.append(match_entry)

        if not matches:
            return "[search_solution] No BM25 matches found for '{}' across candidate solutions".format(query)
        return matches

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


SUMMARY_PROMPT_TEMPLATE = """You are summarizing one candidate solution attempt for a math problem.

Your summary must:
1. Preserve the main mathematical reasoning steps in order.
2. Include all important equations, substitutions, case distinctions, and arithmetic checks.
3. Identify the final answer claimed by the candidate.
4. Note any obvious gap, contradiction, unsupported assumption, or arithmetic/algebra error if present.
5. Be self-contained enough that another model can judge this candidate without reading the full reasoning.

Output exactly in this format:
Reasoning: <brief explanation of how you selected and compressed the important information>
Summary: <the final compact, self-contained summary>

Rules:
- Do not mention the user, the prompt, the task, or that you are summarizing.
- Do not include markdown fences, headings, or any text before "Reasoning:".
- Put only the candidate solution summary after "Summary:".

Summarize this candidate solution in a compact but complete way.

Problem:
{question}

Candidate extracted answer:
{answer}

Full candidate reasoning:
{reasoning}
""".strip()


def parse_summary_response(text: str) -> Optional[str]:
    payload = _strip_markdown_fence(text.strip())
    match = re.search(r"(?im)^\s*Summary\s*:\s*", payload)
    if match is None:
        return None
    summary = payload[match.end() :].strip()
    return summary or None


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


@register_tool("get_solution_summary", allow_overwrite=True)
class GetSolutionSummaryTool(BaseTool):
    name = "get_solution_summary"
    description = "Summarizes one complete candidate solution by calling the configured model endpoint."
    parameters = {
        "type": "object",
        "properties": {
            "solution_id": {"type": "integer", "description": "0-based solution index to summarize."}
        },
        "required": ["solution_id"],
    }

    def __init__(
        self,
        cfg: Optional[dict] = None,
        model: str = "",
        api_base: str = "",
        llm_kwargs: Optional[dict] = None,
        summary_generation_kwargs: Optional[dict] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
    ):
        super().__init__(cfg)
        self.model = model
        self.api_base = api_base
        self.llm_kwargs = llm_kwargs or {}
        self.summary_generation_kwargs = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
        if summary_generation_kwargs:
            self.summary_generation_kwargs.update(summary_generation_kwargs)
        self.completion_fn = completion_fn

    def call(self, params: Union[str, dict], **kwargs) -> Union[dict, str]:
        try:
            solution_id = params["solution_id"]
        except Exception:
            return "[get_solution_summary] Invalid request format: Input must be a JSON object containing 'solution_id'"

        solutions = kwargs.get("solutions", [])
        validated = _validate_solution_id(solution_id, solutions, self.name)
        if isinstance(validated, str):
            return validated
        solution = solutions[validated]
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            question=kwargs.get("question", ""),
            answer=solution.get("answer"),
            reasoning=solution.get("reasoning", ""),
        )
        response = self._completion(messages=[{"role": "user", "content": prompt}])
        raw_summary = _extract_message_content(response).strip()
        summary = parse_summary_response(raw_summary) or raw_summary
        return {
            "solution_id": solution_id,
            "answer": solution.get("answer"),
            "summary": summary,
        }

    def _completion(self, messages: list[dict]) -> Any:
        generation_kwargs = dict(self.summary_generation_kwargs)
        if self.llm_kwargs:
            request_kwargs = {**self.llm_kwargs, **generation_kwargs, "messages": messages}
            request_kwargs.pop("tools", None)
            request_kwargs.pop("parallel_tool_calls", None)
        else:
            request_kwargs = {
                "model": "hosted_vllm/" + self.model,
                "api_key": "EMPTY",
                "api_base": self.api_base,
                "messages": messages,
                **generation_kwargs,
            }
            if "gpt" in self.model.lower() and "oss" not in self.model.lower():
                request_kwargs["model"] = "openai/" + self.model
                request_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
                request_kwargs.pop("api_base", None)

        completion_fn = self.completion_fn
        if completion_fn is None:
            import litellm

            completion_fn = litellm.completion
        return completion_fn(**request_kwargs)

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }


def _extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    if message is None:
        return ""
    for key in ("content", "reasoning_content", "reasoning"):
        value = getattr(message, key, None)
        if value is None and isinstance(message, dict):
            value = message.get(key)
        if value:
            return str(value)
    return ""


@register_tool("finish", allow_overwrite=True)
class FinishTool(BaseTool):
    name = "finish"

    _PROPERTIES = {
        "solution": {"type": "string", "description": "A comprehensive, standalone math solution. First provide a clear step-by-step reasoning process, then give the final answer. The final answer must be placed within \\boxed{}."},
        "reason": {"type": "string", "description": "Meta-reasoning explaining how you evaluated the candidate solutions, what reasoning you relied on, and how you resolved conflicts or inconsistencies."},
    }

    def __init__(self, cfg: Optional[dict] = None, variant: str = "", model: str = ""):
        super().__init__(cfg)
        required = ["solution", "reason"]
        self.description = "Submits the final synthesized math solution."
        properties = {k: v for k, v in self._PROPERTIES.items() if k in required}
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        required = self.parameters["required"]
        missing = [f for f in required if f not in params]
        if missing:
            return f"[finish] Invalid request format: missing field(s): {', '.join(missing)}"
        solution = params.get("solution", "")
        if not isinstance(solution, str) or not solution.strip():
            return "[finish] Invalid solution format: 'solution' must not be empty."
        if _extract_boxed_answer(solution) is None:
            return "[finish] Invalid solution format: final answer must be placed within \\boxed{}."
        return {f: params[f] for f in required}

    def get_tool_definitions(self):
        parameters = self.parameters.copy()
        parameters["additionalProperties"] = False
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
            "strict": True,
        }
