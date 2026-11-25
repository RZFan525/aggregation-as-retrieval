"""OpenAI-compatible API helpers for Qwen3.5 rollout."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..utils.answer_extraction import extract_boxed_answer


QWEN35_MODEL_ALIASES = {
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


@dataclass
class GenerationConfig:
    thinking: Optional[bool]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    presence_penalty: Optional[float]
    min_p: float
    repetition_penalty: float
    seed: Optional[int] = None


@dataclass
class ParsedChatCompletion:
    choices: List["ParsedChatChoice"]
    usage: Dict[str, Any]
    raw_response: Dict[str, Any]


@dataclass
class ParsedChatChoice:
    text: str
    finish_reason: Optional[str]


class OpenAICompatClient:
    """Tiny stdlib-only OpenAI-compatible client for local Qwen3.5 servers."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout_seconds: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
        n: int = 1,
        max_retries: int = 5,
        timeout_seconds: Optional[int] = None,
        include_reasoning: bool = True,
    ) -> ParsedChatCompletion:
        payload = self._build_payload(model=model, messages=messages, config=config, n=n)
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        raw_response = self._post_json_with_retry(
            "/chat/completions", payload, effective_timeout, max_retries
        )
        return parse_chat_completion(raw_response, include_reasoning=include_reasoning)

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        config: GenerationConfig,
        n: int,
    ) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {
            "min_p": config.min_p,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(config.thinking)}
        if config.top_k is not None:
            extra_body["top_k"] = config.top_k

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": n,
            "extra_body": extra_body,
        }
        if config.presence_penalty is not None:
            payload["presence_penalty"] = config.presence_penalty
        if config.seed is not None:
            payload["seed"] = config.seed
        return payload

    def _post_json_with_retry(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout_seconds: int,
        max_retries: int,
    ) -> Dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._post_json(path, payload, timeout_seconds)
            except _TransientHTTPError as exc:
                if attempt >= max_retries:
                    raise RuntimeError(str(exc)) from exc
                backoff = min(60.0, 2.0 ** attempt)
                time.sleep(backoff)
                attempt += 1

    def _post_json(
        self,
        path: str,
        payload: Dict[str, Any],
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        url = "{}{}".format(self.base_url, path)
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.api_key),
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = "HTTP {} for {}: {}".format(exc.code, url, body)
            if 500 <= exc.code < 600 or exc.code == 429:
                raise _TransientHTTPError(message) from exc
            raise RuntimeError(message) from exc
        except urllib_error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout) or isinstance(exc.reason, (ConnectionResetError, ConnectionRefusedError)):
                raise _TransientHTTPError("Transient URL error for {}: {}".format(url, exc)) from exc
            raise _TransientHTTPError("Failed to reach {}: {}".format(url, exc)) from exc
        except socket.timeout as exc:
            raise _TransientHTTPError("Socket timeout for {}: {}".format(url, exc)) from exc


class _TransientHTTPError(Exception):
    pass


def normalize_model_name(model: str) -> str:
    key = model.strip().lower()
    return QWEN35_MODEL_ALIASES.get(key, model)


def default_generation_config(
    thinking: Optional[bool],
    max_tokens: int = 65536,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = 20,
    presence_penalty: Optional[float] = None,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
) -> GenerationConfig:
    return GenerationConfig(
        thinking=thinking,
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else 1.0,
        top_p=top_p if top_p is not None else 0.95,
        top_k=top_k,
        presence_penalty=presence_penalty,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )


def parse_chat_completion(
    payload: Dict[str, Any],
    include_reasoning: bool = True,
) -> ParsedChatCompletion:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("No choices returned by chat completion: {}".format(payload))

    return ParsedChatCompletion(
        choices=[
            _parse_chat_choice(choice, include_reasoning=include_reasoning)
            for choice in choices
        ],
        usage=payload.get("usage") or {},
        raw_response=payload,
    )


def _parse_chat_choice(
    choice: Dict[str, Any],
    include_reasoning: bool = True,
) -> ParsedChatChoice:
    message = choice.get("message") or {}
    content = _flatten_content(message.get("content")).strip()
    if include_reasoning:
        reasoning = _flatten_content(message.get("reasoning_content")).strip()
        if not reasoning:
            reasoning = _flatten_content(message.get("reasoning")).strip()
        parts = [p for p in (reasoning, content) if p]
        text = "\n\n".join(parts)
    else:
        text = content
    return ParsedChatChoice(text=text, finish_reason=choice.get("finish_reason"))


def _flatten_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif "text" in item:
                    chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(value)
