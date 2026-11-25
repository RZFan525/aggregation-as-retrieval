# Modified from AggAgent (https://github.com/princeton-pli/AggAgent, commit 66b697d) to aggregate math solutions; see UPSTREAM.md.
"""AggAgent: tool-calling agent for math solution aggregation."""

from __future__ import annotations

import json
import errno
import os
import random
import re
import time
from importlib.metadata import PackageNotFoundError

from .tools import (
    GetAnswerTool, GetReasoningStepTool, GetSolutionSummaryTool,
    SearchSolutionTool, SearchAllSolutionsBM25Tool, FinishTool,
    build_solution_environment,
    format_metadata, _count_tokens_approx,
)
from .prompts import (
    SYSTEM_PROMPT_AGGAGENT, USER_PROMPT_AGGAGENT,
    SYSTEM_PROMPT_AGGAGENT_QWEN,
    SYSTEM_PROMPT_AGGAGENT_REPORT, USER_PROMPT_AGGAGENT_REPORT,
    FINAL_MESSAGE,
    adapt_search_tool_prompt,
)

LONG_FORM_TASKS = {"healthbench", "researchrubrics"}
FATAL_INFRASTRUCTURE_ERROR_TYPES = (ImportError, FileNotFoundError, PackageNotFoundError)

QWEN35_TOKENIZER_ALIASES = {
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


class AggAgent:
    """
    AggAgent: agentic aggregation over a set of parallel math solutions.

    Usage::

        agent = AggAgent(model="GLM-4-Flash", api_base="http://...", task="browsecomp")
        result = agent.run(question="...", solutions=[{"reasoning": "...", "answer": "..."}])
        # {"solution": "...", "reason": "..."}
        # or on failure: {"solution": None, "reason": None, "error": "..."}
    """

    def __init__(
        self,
        model: str = "",
        api_base: str | None = None,
        max_context_tokens: int = 100 * 1024,
        task: str = "",
        llm_kwargs: dict | None = None,
        solution_step_tokens: int = 1024,
        tokenizer_name: str | None = None,
        tokenizer=None,
        summary_generation_kwargs: dict | None = None,
        summary_completion_fn=None,
        search_tool: str = "rouge",
    ):
        self.model = model
        self.api_base = api_base or ""
        self.max_context_tokens = max_context_tokens
        self.task = task
        self.llm_kwargs = llm_kwargs or {}
        self.solution_step_tokens = solution_step_tokens
        self.tokenizer_name = tokenizer_name or self._default_tokenizer_name(model)
        self.tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer(self.tokenizer_name)
        self.search_tool = self._normalize_search_tool(search_tool)

        variant = "long_form" if task in LONG_FORM_TASKS else ""
        tools = [
            GetAnswerTool(),
            self._make_search_tool(self.search_tool),
            GetReasoningStepTool(),
            GetSolutionSummaryTool(
                model=model,
                api_base=self.api_base,
                llm_kwargs=self.llm_kwargs,
                summary_generation_kwargs=summary_generation_kwargs,
                completion_fn=summary_completion_fn,
            ),
            FinishTool(variant=variant, model=model),
        ]

        self.tool_map = {tool.name: tool for tool in tools}
        self.tool_description = [self.tool_map[t].get_tool_definitions() for t in self.tool_map]

    def run(self, question: str, solutions: list[dict]) -> dict:
        """
        Synthesize a final answer from N parallel candidate math solutions.

        Args:
            question: The task/question string.
            solutions: List of N candidate solution dicts with keys such as
                       "reasoning" and "answer". Message-log dicts are also
                       accepted as a compatibility fallback.

        Returns:
            On success: {"solution": str, "reason": str}
            On failure: {"solution": None, "reason": None, "error": str}
        """
        output = self._run(question, solutions)
        result = output.get("result")
        if result is None or not isinstance(result, dict) or "solution" not in result:
            return {"solution": None, "reason": None, "error": "Agent did not produce a valid solution"}
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_tokenizer_name(model: str) -> str:
        key = (model or "").strip().lower()
        if key in QWEN35_TOKENIZER_ALIASES:
            return QWEN35_TOKENIZER_ALIASES[key]
        if "qwen3.5" in key:
            return model
        return "Qwen/Qwen3.5-9B"

    @staticmethod
    def _load_tokenizer(tokenizer_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for AggMath solution chunking with the "
                "Qwen3.5 tokenizer. Pass tokenizer=... in tests or install transformers."
            ) from exc
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    @staticmethod
    def _normalize_search_tool(search_tool: str) -> str:
        normalized = (search_tool or "rouge").strip().lower()
        aliases = {
            "rouge": "rouge",
            "rougel": "rouge",
            "rouge_l": "rouge",
            "bm25": "bm25",
            "bm25s": "bm25",
            "global_bm25": "bm25",
        }
        if normalized not in aliases:
            raise ValueError("Unsupported search_tool '{}'. Expected 'rouge' or 'bm25'.".format(search_tool))
        return aliases[normalized]

    @staticmethod
    def _make_search_tool(search_tool: str):
        if search_tool == "bm25":
            return SearchAllSolutionsBM25Tool()
        return SearchSolutionTool()

    def _adapt_search_tool_prompt(self, system_prompt: str) -> str:
        return adapt_search_tool_prompt(system_prompt, self.search_tool)

    def _prepare_solutions(self, run_results: list[dict]) -> list[dict]:
        return build_solution_environment(
            run_results,
            tokenizer=self.tokenizer,
            step_token_target=self.solution_step_tokens,
        )

    def count_tokens_approx(self, messages, chars_per_token=4.0):
        tool_chars = len(json.dumps(self.tool_description, ensure_ascii=False))
        return _count_tokens_approx(messages, chars_per_token) + int(tool_chars / chars_per_token)

    def sanitize_tool_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '', name)

    def _message_to_dict(self, assistant_message):
        content = assistant_message.get("content") or ""
        reasoning_content = assistant_message.get("reasoning_content") or ""
        tool_calls_list = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": self.sanitize_tool_name(tc.function.name),
                    "arguments": tc.function.arguments,
                }
            }
            for tc in assistant_message.tool_calls[:1]
        ] if assistant_message.tool_calls else []
        if "gemini" in self.model:
            thinking_blocks = assistant_message.get("thinking_blocks") or []
            return {"role": "assistant", "content": content, "reasoning_content": reasoning_content,
                    "thinking_blocks": thinking_blocks, "tool_calls": tool_calls_list}
        else:
            return {"role": "assistant", "content": content, "reasoning": reasoning_content,
                    "tool_calls": tool_calls_list}

    def call_server(self, messages, max_tries=5, only_finish=False):
        base_sleep_time = 1
        tools = self.tool_description if not only_finish else [self.tool_map["finish"].get_tool_definitions()]
        if self.llm_kwargs:
            litellm_kwargs = {**self.llm_kwargs, "messages": messages,"tools": tools}
        else:
            litellm_kwargs = {
                "model": "hosted_vllm/" + self.model,
                "api_key": "EMPTY",
                "api_base": self.api_base,
                "messages": messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 10000,
                "parallel_tool_calls": False,
                "tools": tools,
            }
            if "gemini" in self.model.lower():
                litellm_kwargs["model"] = "gemini/" + self.model
                litellm_kwargs["api_key"] = os.getenv("GEMINI_API_KEY")
                del litellm_kwargs["api_base"]
                litellm_kwargs["reasoning_effort"] = "low"
                del litellm_kwargs["parallel_tool_calls"]
            elif "oss" in self.model.lower():
                litellm_kwargs["reasoning_effort"] = "high"
            elif "gpt" in self.model.lower():
                litellm_kwargs["model"] = "openai/" + self.model
                litellm_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
                del litellm_kwargs["api_base"]
                del litellm_kwargs["top_p"]
            elif "minimax" in self.model.lower():
                litellm_kwargs["extra_body"] = {"reasoning_split": True}

        for attempt in range(max_tries):
            try:
                import litellm

                response = litellm.completion(**litellm_kwargs)
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return response
            except Exception as e:
                if e.__class__.__name__ == "BadRequestError" and (
                    "context length" in str(e).lower() or "input tokens" in str(e).lower()
                ):
                    print(f"Context length exceeded: {e}")
                    return "ContextLengthError"
                print(f"Error: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                time.sleep(sleep_time)

        return "Server error"

    def _make_output(self, result, messages, stats):
        return {"result": result, "messages": messages, "stats": stats}

    def _run(self, question, run_results):
        solutions = self._prepare_solutions(run_results)
        tool_state = {}
        metadata = format_metadata(solutions)
        iteration = 0
        MAX_ITERATIONS = 100

        if self.task in LONG_FORM_TASKS:
            system_prompt = SYSTEM_PROMPT_AGGAGENT_REPORT
            user_prompt = USER_PROMPT_AGGAGENT_REPORT.format(
                question=question,
                metadata=metadata,
                solution_N=len(solutions),
                solution_N_minus_one=max(len(solutions) - 1, 0),
            )
        elif "qwen" in self.model.lower():
            system_prompt = SYSTEM_PROMPT_AGGAGENT_QWEN
            user_prompt = USER_PROMPT_AGGAGENT.format(
                question=question,
                metadata=metadata,
                solution_N=len(solutions),
                solution_N_minus_one=max(len(solutions) - 1, 0),
            )
        else:
            system_prompt = SYSTEM_PROMPT_AGGAGENT
            user_prompt = USER_PROMPT_AGGAGENT.format(
                question=question,
                metadata=metadata,
                solution_N=len(solutions),
                solution_N_minus_one=max(len(solutions) - 1, 0),
            )

        system_prompt = self._adapt_search_tool_prompt(system_prompt)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        full_messages = list(messages)

        stats = {
            "iterations": 0,
            "server_errors": 0,
            "tool_call_errors": 0,
            "tool_calls": {},
            "context_limit_reached": False,
            "token_usage_each_step": [],
        }

        def store_token_usage(response, iteration):
            if response is None or isinstance(response, str):
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            elif hasattr(response, "usage") and response.usage:
                entry = {
                    "iteration": iteration,
                    "input_tokens": getattr(response.usage, "prompt_tokens", None),
                    "output_tokens": getattr(response.usage, "completion_tokens", None),
                }
            else:
                entry = {"iteration": iteration, "input_tokens": None, "output_tokens": None}
            stats["token_usage_each_step"].append(entry)

        while iteration < MAX_ITERATIONS:
            iteration += 1
            stats["iterations"] = iteration
            response = self.call_server(messages)
            store_token_usage(response, iteration)

            if response == "ContextLengthError":
                stats["context_limit_reached"] = True
                return self._make_output(None, full_messages, stats)

            if isinstance(response, str):
                stats["server_errors"] += 1
                error_message = {"role": "assistant", "content": response}
                messages.append(error_message)
                full_messages.append(error_message)
                continue

            assistant_message = response.choices[0].message
            assistant_dict = self._message_to_dict(assistant_message)
            messages.append(assistant_dict)
            full_messages.append(assistant_dict)

            for tool_call in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                tool_name = self.sanitize_tool_name(tool_call.function.name)
                stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    print(e)
                    stats["tool_call_errors"] += 1
                    tool_result = None
                    tool_result_str = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'
                else:
                    try:
                        tool_result = self.custom_call_tool(
                            tool_name,
                            tool_args,
                            solutions=solutions,
                            question=question,
                            tool_state=tool_state,
                        )
                        tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                    except Exception as e:
                        if _is_fatal_infrastructure_error(e):
                            raise
                        print(e)
                        stats["tool_call_errors"] += 1
                        tool_result = None
                        tool_result_str = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field'

                tool_message = {"role": "tool", "tool_call_id": tool_call.id, "name": tool_name,
                                "content": tool_result_str}
                messages.append(tool_message)
                full_messages.append(tool_message)
                if tool_name == "finish" and isinstance(tool_result, dict):
                    return self._make_output(tool_result, full_messages, stats)

            if self.count_tokens_approx(messages) > self.max_context_tokens:
                stats["context_limit_reached"] = True
                final_msg = {"role": "user", "content": FINAL_MESSAGE}
                messages.append(final_msg)
                full_messages.append(final_msg)
                response = self.call_server(messages, only_finish=True)
                store_token_usage(response, iteration)
                if isinstance(response, str):
                    stats["server_errors"] += 1
                    err_msg = {"role": "assistant", "content": response}
                    messages.append(err_msg)
                    full_messages.append(err_msg)
                    return self._make_output(None, full_messages, stats)

                assistant_message = response.choices[0].message
                assistant_dict2 = self._message_to_dict(assistant_message)
                messages.append(assistant_dict2)
                full_messages.append(assistant_dict2)
                for tc in assistant_message.tool_calls[:1] if assistant_message.tool_calls else []:
                    tn = self.sanitize_tool_name(tc.function.name)
                    stats["tool_calls"][tn] = stats["tool_calls"].get(tn, 0) + 1
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except Exception as e:
                        print(e)
                        stats["tool_call_errors"] += 1
                        tool_result = None
                        tool_result_str = "Error: Tool call is not a valid JSON."
                    else:
                        try:
                            tool_result = self.custom_call_tool(
                                tn,
                                tool_args,
                                solutions=solutions,
                                question=question,
                                tool_state=tool_state,
                            )
                            tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                        except Exception as e:
                            if _is_fatal_infrastructure_error(e):
                                raise
                            print(e)
                            stats["tool_call_errors"] += 1
                            tool_result = None
                            tool_result_str = "Error: Tool call is not a valid JSON."
                    tool_message = {"role": "tool", "tool_call_id": tc.id, "name": tn,
                                    "content": tool_result_str}
                    messages.append(tool_message)
                    full_messages.append(tool_message)
                    if tn == "finish" and isinstance(tool_result, dict):
                        return self._make_output(tool_result, full_messages, stats)
                return self._make_output(None, full_messages, stats)

        return self._make_output(None, full_messages, stats)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            return self.tool_map[tool_name].call(tool_args, **kwargs)
        return f"Error: Tool {tool_name} not found"


def _is_fatal_infrastructure_error(exc: BaseException) -> bool:
    if isinstance(exc, FATAL_INFRASTRUCTURE_ERROR_TYPES):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOENT
