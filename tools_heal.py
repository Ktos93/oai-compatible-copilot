# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
llama-server inference backend for GGUF models.

Manages a llama-server subprocess and proxies chat completions
through its OpenAI-compatible /v1/chat/completions endpoint.
"""

import atexit
import contextlib
import json
import re
import struct
import structlog
from loggers import get_logger
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator, List, Optional
from urllib.parse import urlparse

import httpx

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


# ── Pre-compiled patterns for plan-without-action re-prompt ──
# Forward-looking intent signals that indicate the model is
# describing what it *will* do rather than giving a final answer.
_INTENT_SIGNAL = re.compile(
    r"(?i)("
    # Direct intent: "I'll ...", "I will ...", "Let me ...", "I am going to ..."
    # Handles both straight and curly apostrophes.
    # Excludes "I can", "I should", "I want to", "let's" which
    # appear frequently in direct answers / explanations.
    r"\b(i['\u2019](ll|m going to|m gonna)|i am (going to|gonna)|i will|i shall|let me|allow me)\b"
    r"|"
    # Step/plan framing: "First ...", "Step 1:", "Here's my plan"
    r"\b(?:first\b|step \d+:?|here['\u2019]?s (?:my |the |a )?(?:plan|approach))"
    r"|"
    # "Now I" / "Next I" patterns
    r"\b(?:now i|next i)\b"
    r")"
)
_MAX_REPROMPTS = 3

# Without max_tokens, llama-server defaults to n_predict = n_ctx (up to
# 262144 for Qwen3.5), producing many-minute zombie decodes when cancel
# fails. t_max_predict_ms is a wall-clock backstop applied unconditionally,
# but the llama.cpp README notes it ONLY fires after a newline has been
# generated -- a model stuck in a long unbroken non-newline sequence is
# unbounded by it. So we still want a token cap as the front-line limiter.
#
# The cap is the model's effective context length when we know it,
# falling back to a generous floor when metadata is unavailable. 4096 was
# too low: Qwen3 / gpt-oss reasoning traces routinely exceed it, and any
# OpenAI-API caller that omits max_tokens (langchain, llama-index, raw
# curl) sees responses silently truncated mid-sentence.
_DEFAULT_MAX_TOKENS_FLOOR = 32768
_DEFAULT_T_MAX_PREDICT_MS = 600_000  # 10 min
_REPROMPT_MAX_CHARS = 2000

# ── Pre-compiled patterns for GGUF shard detection ───────────
_SHARD_FULL_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
_SHARD_RE = re.compile(r"^(.*)-\d{5}-of-\d{5}\.gguf$")


# Model size extraction — lazy import to avoid pulling in transformers
# at module level.  See PR description for the full explanation.
def _extract_model_size_b(model_id: str):
    from utils.models import extract_model_size_b

    return extract_model_size_b(model_id)


# ── Pre-compiled patterns for tool XML stripping ─────────────
_TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
]
_TOOL_ALL_PATS = _TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
]

# ── Pre-compiled patterns for tool-call XML parsing ──────────
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")
_TC_END_TAG_RE = re.compile(r"</tool_call>")
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")


_TOOL_TEMPLATE_MARKERS = (
    "{%- if tools %}",
    "{%- if tools -%}",
    "{% if tools %}",
    "{% if tools -%}",
    '"role" == "tool"',
    "'role' == 'tool'",
    'message.role == "tool"',
    "message.role == 'tool'",
)

    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _parse_tool_calls_from_text(content: str) -> list[dict]:
        """
        Parse tool calls from XML markup in content text.

        Handles formats like:
          <tool_call>{"name":"web_search","arguments":{"query":"..."}}</tool_call>
          <tool_call><function=web_search><parameter=query>...</parameter></function></tool_call>
        Closing tags (</tool_call>, </function>, </parameter>) are all optional
        since models frequently omit them.
        """
        tool_calls = []

        # Pattern 1: JSON inside <tool_call> tags.
        # Use balanced-brace extraction that skips braces inside JSON strings.
        for m in _TC_JSON_START_RE.finditer(content):
            brace_start = m.end() - 1  # position of the opening {
            depth, i = 0, brace_start
            in_string = False
            while i < len(content):
                ch = content[i]
                if in_string:
                    if ch == "\\" and i + 1 < len(content):
                        i += 2  # skip escaped character
                        continue
                    if ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                json_str = content[brace_start : i + 1]
                try:
                    obj = json.loads(json_str)
                    tc = {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": obj.get("name", ""),
                            "arguments": obj.get("arguments", {}),
                        },
                    }
                    if isinstance(tc["function"]["arguments"], dict):
                        tc["function"]["arguments"] = json.dumps(
                            tc["function"]["arguments"]
                        )
                    tool_calls.append(tc)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Pattern 2: XML-style <function=name><parameter=key>value</parameter></function>
        # All closing tags optional -- models frequently omit </parameter>,
        # </function>, and/or </tool_call>.
        if not tool_calls:
            # Step 1: Find all <function=name> positions and extract their bodies.
            # Body boundary: use only </tool_call> or next <function= as hard
            # boundaries.  We avoid using </function> as a boundary because
            # code parameter values can contain that literal string.
            # After extracting, we trim a trailing </function> if present.
            func_starts = list(_TC_FUNC_START_RE.finditer(content))
            for idx, fm in enumerate(func_starts):
                func_name = fm.group(1)
                body_start = fm.end()
                # Hard boundaries: next <function= tag or </tool_call>
                next_func = (
                    func_starts[idx + 1].start()
                    if idx + 1 < len(func_starts)
                    else len(content)
                )
                end_tag = _TC_END_TAG_RE.search(content[body_start:])
                if end_tag:
                    body_end = body_start + end_tag.start()
                else:
                    body_end = len(content)
                body_end = min(body_end, next_func)
                body = content[body_start:body_end]
                # Trim trailing </function> if present (it's the real closing tag)
                body = _TC_FUNC_CLOSE_RE.sub("", body)

                # Step 2: Extract parameters from body.
                # For single-parameter functions (the common case: code, command,
                # query), use body end as the only boundary to avoid false matches
                # on </parameter> inside code strings.
                arguments = {}
                param_starts = list(_TC_PARAM_START_RE.finditer(body))
                if len(param_starts) == 1:
                    # Single parameter: value is everything from after the tag
                    # to end of body, trimming any trailing </parameter>.
                    pm = param_starts[0]
                    val = body[pm.end() :]
                    val = _TC_PARAM_CLOSE_RE.sub("", val)
                    arguments[pm.group(1)] = val.strip()
                else:
                    for pidx, pm in enumerate(param_starts):
                        param_name = pm.group(1)
                        val_start = pm.end()
                        # Value ends at next <parameter= or end of body
                        next_param = (
                            param_starts[pidx + 1].start()
                            if pidx + 1 < len(param_starts)
                            else len(body)
                        )
                        val = body[val_start:next_param]
                        # Trim trailing </parameter> if present
                        val = _TC_PARAM_CLOSE_RE.sub("", val)
                        arguments[param_name] = val.strip()

                tc = {
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(arguments),
                    },
                }
                tool_calls.append(tc)

        return tool_calls

    @staticmethod
    def _build_openai_messages(
        messages: list[dict],
        image_b64: Optional[str] = None,
    ) -> list[dict]:
        """
        Build OpenAI-format messages, optionally injecting an image_url
        content part into the last user message for vision models.

        If no image is provided, returns messages as-is.
        """
        if not image_b64:
            return messages

        # Find the last user message and convert to multimodal content parts
        result = [msg.copy() for msg in messages]
        last_user_idx = None
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx is not None:
            text_content = result[last_user_idx].get("content", "")
            result[last_user_idx]["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]

        return result

    # ── Generation (proxy to llama-server) ────────────────────────

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """Iterate over an httpx streaming response with cancel support.

        Checks cancel_event between chunks and on ReadTimeout.  The
        cancel watcher in _stream_with_retry also calls response.close()
        on cancel, which unblocks iter_text() once the response exists.
        During normal streaming llama-server sends tokens frequently,
        so the cancel check between chunks is the primary mechanism.
        """
        text_iter = response.iter_text()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                chunk = next(text_iter)
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                # No data within the timeout window -- just loop back
                # and re-check cancel_event.
                continue

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
        headers: Optional[dict] = None,
    ):
        """Open an httpx streaming POST with cancel support.

        Sends the request once with a long read timeout (120 s) so
        prompt processing (prefill) can finish without triggering a
        retry storm.  The previous 0.5 s timeout caused duplicate POST
        requests every half second, forcing llama-server to restart
        processing each time.

        A background watcher thread provides cancel by closing the
        response when cancel_event is set.  Limitation: httpx does not
        allow interrupting a blocked read from another thread before
        the response object exists, so cancel during the initial
        header wait (prefill phase) only takes effect once headers
        arrive.  After that, response.close() unblocks reads promptly.
        In practice llama-server prefill is 1-5 s for typical prompts,
        during which cancel is deferred -- still much better than the
        old retry storm which made prefill slower.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise GeneratorExit

        # Background watcher: close the response if cancel is requested.
        # Only effective after response headers arrive (httpx limitation).
        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    # Cancel requested. Keep polling until the response object
                    # exists so we can close it, or until the main thread
                    # finishes on its own (_cancel_closed is set in finally).
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        if r is not None:
                            try:
                                r.close()
                                return
                            except Exception as e:
                                logger.debug(
                                    f"Error closing response in cancel watcher: {e}"
                                )
                        # Response not created yet -- wait briefly and retry
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(
                target = _cancel_watcher, daemon = True, name = "prefill-cancel"
            )
            watcher.start()

        try:
            # Long read timeout so prefill (prompt processing) can finish
            # without triggering a retry storm.  Cancel during both
            # prefill and streaming is handled by the watcher thread
            # which closes the response, unblocking any httpx read.
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = 120.0,
                write = 10,
                pool = 10,
            )
            with client.stream(
                "POST",
                url,
                json = payload,
                timeout = prefill_timeout,
                headers = headers,
            ) as response:
                _response_ref[0] = response
                if cancel_event is not None and cancel_event.is_set():
                    raise GeneratorExit
                yield response
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.CloseError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise GeneratorExit
            raise
        finally:
            _cancel_closed.set()

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> Generator[str | dict, None, None]:
        """
        Send a chat completion request to llama-server and stream tokens back.

        Uses /v1/chat/completions — llama-server handles chat template
        application and vision (multimodal image_url parts) natively.

        Yields cumulative text (matching InferenceBackend's convention).
        """
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        openai_messages = self._build_openai_messages(messages, image_b64)

        payload = {
            "messages": openai_messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        # Pass enable_thinking / reasoning_effort / preserve_thinking per-request
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        # Default cap to the model's effective context length when known,
        # otherwise the conservative floor. The wall-clock backstop below
        # keeps a stuck model from running indefinitely either way.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
        if stop:
            payload["stop"] = stop
        payload["stream_options"] = {"include_usage": True}

        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None

        try:
            # _stream_with_retry uses a 120 s read timeout so prefill
            # can finish.  Cancel during streaming is handled by the
            # watcher thread (closes the response on cancel_event).
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            _auth_headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            )
            with httpx.Client(
                timeout = stream_timeout, limits = httpx.Limits(max_keepalive_connections = 0)
            ) as client:
                with self._stream_with_retry(
                    client,
                    url,
                    payload,
                    cancel_event,
                    headers = _auth_headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    has_content_tokens = False
                    reasoning_text = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if has_content_tokens:
                                        # Real thinking + content: close the tag
                                        cumulative += "</think>"
                                        yield cumulative
                                    else:
                                        # Only reasoning_content, no content tokens:
                                        # the model put its entire reply in reasoning
                                        # (e.g. Qwen3 always-think mode). Show it
                                        # as the main response, not as a thinking block.
                                        cumulative = reasoning_text
                                        yield cumulative
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                data = json.loads(line[6:])
                                # Capture server timings/usage from final chunks
                                _chunk_timings = data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    # Handle reasoning/thinking tokens
                                    # llama-server sends these as "reasoning_content"
                                    # Wrap in <think> tags for the frontend parser
                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield cumulative

                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                        cumulative += token
                                        yield cumulative
                            except json.JSONDecodeError:
                                logger.debug(
                                    f"Skipping malformed SSE line: {line[:100]}"
                                )
                        if _stream_done:
                            break  # exit outer for
                    if _metadata_usage or _metadata_timings:
                        yield {
                            "type": "metadata",
                            "usage": _metadata_usage,
                            "timings": _metadata_timings,
                        }

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise

    # ── Tool-calling agentic loop ──────────────────────────────

    def generate_chat_completion_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        max_tool_iterations: int = 25,
        auto_heal_tool_calls: bool = True,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """
        Agentic loop: let the model call tools, execute them, and continue.

        Yields dicts with:
          {"type": "status", "text": "Searching: ..."/"Reading: ..."}   -- tool status updates
          {"type": "content", "text": "token"}            -- streamed content tokens (cumulative)
          {"type": "reasoning", "text": "token"}          -- streamed reasoning tokens (cumulative)
        """
        from core.inference.tools import execute_tool

        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")

        conversation = list(messages)
        url = f"{self.base_url}/v1/chat/completions"
        _accumulated_completion_tokens = 0
        _accumulated_predicted_ms = 0.0
        _accumulated_predicted_n = 0

        def _strip_tool_markup(text: str, *, final: bool = False) -> str:
            if not auto_heal_tool_calls:
                return text
            patterns = _TOOL_ALL_PATS if final else _TOOL_CLOSED_PATS
            for pat in patterns:
                text = pat.sub("", text)
            return text.strip() if final else text

        # XML prefixes that signal a tool call in content.
        # Empty when auto_heal is disabled so the buffer never
        # speculatively holds content for XML detection.
        _TOOL_XML_SIGNALS = (
            ("<tool_call>", "<function=") if auto_heal_tool_calls else ()
        )
        _MAX_BUFFER_CHARS = 32

        # ── Duplicate tool-call detection ────────────────────────
        # Track recent (tool_name, arguments) hashes to detect loops
        # where the model repeats the exact same call.  Retries after
        # a transient failure are allowed (only block when the previous
        # identical call succeeded).
        _tool_call_history: list[tuple[str, bool]] = []  # (key, failed)

        # ── Re-prompt on plan-without-action ─────────────────
        # When the model describes what it intends to do (forward-looking
        # language) without actually calling a tool, re-prompt once.
        # Only triggers on responses that signal intent/planning -- a
        # direct answer like "4" or "Hello!" will not match.
        # Pattern is compiled once at module level (_INTENT_SIGNAL).
        _reprompt_count = 0

        # Reserve extra iterations for re-prompts so they don't
        # consume the caller's tool-call budget.  Only add the
        # extra slot when tool iterations are actually allowed.
        _extra = _MAX_REPROMPTS if max_tool_iterations > 0 else 0
        for iteration in range(max_tool_iterations + _extra):
            if cancel_event is not None and cancel_event.is_set():
                return

            # Build payload -- stream: True so we detect tool signals
            # in the first 1-2 chunks without a non-streaming penalty.
            payload = {
                "messages": conversation,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if top_k >= 0 else 0,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "tools": tools,
                "tool_choice": "auto",
            }
            _reasoning_kw = self._request_reasoning_kwargs(
                enable_thinking, reasoning_effort, preserve_thinking
            )
            if _reasoning_kw is not None:
                payload["chat_template_kwargs"] = _reasoning_kw
            payload["max_tokens"] = (
                max_tokens
                if max_tokens is not None
                else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
            )
            payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
            if stop:
                payload["stop"] = stop

            try:
                _auth_headers = (
                    {"Authorization": f"Bearer {self._api_key}"}
                    if self._api_key
                    else None
                )

                # ── Speculative buffer state machine ──────────────────
                # BUFFERING: accumulating content, checking for tool signals
                # STREAMING: no tool detected, yielding tokens to caller
                # DRAINING:  tool signal found, silently consuming rest
                _S_BUFFERING = 0
                _S_STREAMING = 1
                _S_DRAINING = 2

                detect_state = _S_BUFFERING
                content_buffer = ""  # Raw content held during BUFFERING
                content_accum = ""  # All content tokens (for tool parsing)
                reasoning_accum = ""
                cumulative_display = ""  # Cumulative text yielded (with <think>)
                in_thinking = False
                has_content_tokens = False
                tool_calls_acc = {}  # Structured delta.tool_calls fragments
                has_structured_tc = False
                _iter_usage = None
                _iter_timings = None
                _stream_done = False
                _last_emitted = ""

                stream_timeout = httpx.Timeout(
                    connect = 10,
                    read = 0.5,
                    write = 10,
                    pool = 10,
                )
                with httpx.Client(
                    timeout = stream_timeout,
                    limits = httpx.Limits(max_keepalive_connections = 0),
                ) as client:
                    with self._stream_with_retry(
                        client,
                        url,
                        payload,
                        cancel_event,
                        headers = _auth_headers,
                    ) as response:
                        if response.status_code != 200:
                            error_body = response.read().decode()
                            raise RuntimeError(
                                f"llama-server returned {response.status_code}: "
                                f"{error_body}"
                            )

                        raw_buf = ""
                        for raw_chunk in self._iter_text_cancellable(
                            response,
                            cancel_event,
                        ):
                            raw_buf += raw_chunk
                            while "\n" in raw_buf:
                                line, raw_buf = raw_buf.split("\n", 1)
                                line = line.strip()

                                if not line:
                                    continue
                                if line == "data: [DONE]":
                                    # Flush thinking state for STREAMING
                                    if detect_state == _S_STREAMING and in_thinking:
                                        if has_content_tokens:
                                            cumulative_display += "</think>"
                                            yield {
                                                "type": "content",
                                                "text": _strip_tool_markup(
                                                    cumulative_display,
                                                    final = True,
                                                ),
                                            }
                                        else:
                                            cumulative_display = reasoning_accum
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }
                                    _stream_done = True
                                    break  # exit inner while
                                if not line.startswith("data: "):
                                    continue

                                try:
                                    chunk_data = json.loads(line[6:])
                                    _ct = chunk_data.get("timings")
                                    if _ct:
                                        _iter_timings = _ct
                                    _cu = chunk_data.get("usage")
                                    if _cu:
                                        _iter_usage = _cu

                                    choices = chunk_data.get("choices", [])
                                    if not choices:
                                        continue

                                    delta = choices[0].get("delta", {})

                                    # ── Structured tool_calls ──
                                    tc_deltas = delta.get("tool_calls")
                                    if tc_deltas:
                                        # Once visible content has been
                                        # emitted, do not reclassify this
                                        # turn as a tool call.
                                        if _last_emitted:
                                            continue
                                        has_structured_tc = True
                                        detect_state = _S_DRAINING
                                        for tc_d in tc_deltas:
                                            idx = tc_d.get("index", 0)
                                            if idx not in tool_calls_acc:
                                                tool_calls_acc[idx] = {
                                                    "id": tc_d.get("id", f"call_{idx}"),
                                                    "type": "function",
                                                    "function": {
                                                        "name": "",
                                                        "arguments": "",
                                                    },
                                                }
                                            elif tc_d.get("id"):
                                                # Update ID if real one
                                                # arrives on a later delta
                                                tool_calls_acc[idx]["id"] = tc_d["id"]
                                            func = tc_d.get("function", {})
                                            if func.get("name"):
                                                tool_calls_acc[idx]["function"][
                                                    "name"
                                                ] += func["name"]
                                            if func.get("arguments"):
                                                tool_calls_acc[idx]["function"][
                                                    "arguments"
                                                ] += func["arguments"]
                                        continue

                                    # ── Reasoning tokens ──
                                    # Only yield in STREAMING state. In BUFFERING
                                    # and DRAINING, accumulate silently so we don't
                                    # corrupt the consumer's prev_text tracker
                                    # (routes/inference.py never resets prev_text
                                    # between tool iterations).
                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_accum += reasoning
                                        if detect_state == _S_STREAMING:
                                            if not in_thinking:
                                                cumulative_display += "<think>"
                                                in_thinking = True
                                            cumulative_display += reasoning
                                            yield {
                                                "type": "content",
                                                "text": cumulative_display,
                                            }

                                    # ── Content tokens ──
                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        content_accum += token

                                        if detect_state == _S_DRAINING:
                                            pass  # accumulate silently

                                        elif detect_state == _S_STREAMING:
                                            if in_thinking:
                                                cumulative_display += "</think>"
                                                in_thinking = False
                                            cumulative_display += token
                                            cleaned = _strip_tool_markup(
                                                cumulative_display,
                                            )
                                            if len(cleaned) > len(_last_emitted):
                                                _last_emitted = cleaned
                                                yield {
                                                    "type": "content",
                                                    "text": cleaned,
                                                }

                                        elif detect_state == _S_BUFFERING:
                                            content_buffer += token
                                            stripped_buf = content_buffer.lstrip()
                                            if not stripped_buf:
                                                continue

                                            # Check tool signal prefixes
                                            is_prefix = False
                                            is_match = False
                                            for sig in _TOOL_XML_SIGNALS:
                                                if stripped_buf.startswith(sig):
                                                    is_match = True
                                                    break
                                                if sig.startswith(stripped_buf):
                                                    is_prefix = True
                                                    break

                                            if is_match:
                                                detect_state = _S_DRAINING
                                            elif (
                                                is_prefix
                                                and len(stripped_buf)
                                                < _MAX_BUFFER_CHARS
                                            ):
                                                pass  # keep buffering
                                            else:
                                                # Not a tool -- flush buffer
                                                detect_state = _S_STREAMING
                                                # Flush any reasoning accumulated
                                                # during BUFFERING phase
                                                if reasoning_accum:
                                                    cumulative_display += "<think>"
                                                    cumulative_display += (
                                                        reasoning_accum
                                                    )
                                                    cumulative_display += "</think>"
                                                cumulative_display += content_buffer
                                                cleaned = _strip_tool_markup(
                                                    cumulative_display,
                                                )
                                                if len(cleaned) > len(_last_emitted):
                                                    _last_emitted = cleaned
                                                    yield {
                                                        "type": "content",
                                                        "text": cleaned,
                                                    }

                                except json.JSONDecodeError:
                                    logger.debug(
                                        f"Skipping malformed SSE line: " f"{line[:100]}"
                                    )
                            if _stream_done:
                                break  # exit outer for

                # ── Resolve BUFFERING at stream end ──
                if detect_state == _S_BUFFERING:
                    stripped_buf = content_buffer.lstrip()
                    if (
                        stripped_buf
                        and auto_heal_tool_calls
                        and any(s in stripped_buf for s in _TOOL_XML_SIGNALS)
                    ):
                        detect_state = _S_DRAINING
                    elif content_accum or reasoning_accum:
                        detect_state = _S_STREAMING
                        if content_buffer:
                            # Flush any reasoning accumulated first
                            if reasoning_accum:
                                cumulative_display += "<think>"
                                cumulative_display += reasoning_accum
                                cumulative_display += "</think>"
                            cumulative_display += content_buffer
                            yield {
                                "type": "content",
                                "text": _strip_tool_markup(
                                    cumulative_display,
                                    final = True,
                                ),
                            }
                        elif reasoning_accum and not has_content_tokens:
                            # Reasoning-only response (no content tokens):
                            # show reasoning as plain text, matching
                            # the final streaming pass behavior for
                            # models that put everything in reasoning.
                            cumulative_display = reasoning_accum
                            yield {
                                "type": "content",
                                "text": cumulative_display,
                            }
                    else:
                        return

                # ── STREAMING path: no tool call ──
                if detect_state == _S_STREAMING:
                    # Safety net: check for XML tool signals in content.
                    # The route layer resets prev_text on tool_start, so
                    # post-tool synthesis streams correctly even if
                    # content was already emitted before the tool XML.
                    _safety_tc = None
                    if auto_heal_tool_calls and any(
                        s in content_accum for s in _TOOL_XML_SIGNALS
                    ):
                        _safety_tc = self._parse_tool_calls_from_text(
                            content_accum,
                        )
                    if not _safety_tc:
                        # ── Re-prompt on plan-without-action ──
                        # If the model described what it intends to do
                        # (forward-looking language) without calling any
                        # tool, nudge it to act.  Only fires once per
                        # request and only on short responses that
                        # contain intent signals -- a direct answer
                        # like "4" or "Hello!" won't trigger this.
                        # Use content if available, otherwise fall back
                        # to reasoning text (reasoning-only stalls).
                        _stripped = content_accum.strip()
                        if not _stripped:
                            _stripped = reasoning_accum.strip()
                        if (
                            tools
                            and _reprompt_count < _MAX_REPROMPTS
                            and 0 < len(_stripped) < _REPROMPT_MAX_CHARS
                            and _INTENT_SIGNAL.search(_stripped)
                        ):
                            _reprompt_count += 1
                            logger.info(
                                f"Re-prompt {_reprompt_count}/{_MAX_REPROMPTS}: "
                                f"model responded without calling tools "
                                f"({len(_stripped)} chars)"
                            )
                            conversation.append(
                                {
                                    "role": "assistant",
                                    "content": _stripped,
                                }
                            )
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "STOP. Do NOT write code or explain. "
                                        "You MUST call a tool NOW. "
                                        "Call web_search or python immediately."
                                    ),
                                }
                            )
                            # Accumulate tokens and timing from this iteration
                            _fu_r = _iter_usage or {}
                            _accumulated_completion_tokens += _fu_r.get(
                                "completion_tokens", 0
                            )
                            _it_r = _iter_timings or {}
                            _accumulated_predicted_ms += _it_r.get("predicted_ms", 0)
                            _accumulated_predicted_n += _it_r.get("predicted_n", 0)
                            yield {"type": "status", "text": ""}
                            continue

                        # Content was already streamed.  Yield metadata.
                        yield {"type": "status", "text": ""}
                        _fu = _iter_usage or {}
                        _fc = _fu.get("completion_tokens", 0)
                        _fp = _fu.get("prompt_tokens", 0)
                        _tc = _fc + _accumulated_completion_tokens
                        if (
                            _iter_usage
                            or _iter_timings
                            or _accumulated_completion_tokens
                        ):
                            _mt = dict(_iter_timings) if _iter_timings else {}
                            if _accumulated_predicted_ms or _accumulated_predicted_n:
                                _mt["predicted_ms"] = (
                                    _mt.get("predicted_ms", 0)
                                    + _accumulated_predicted_ms
                                )
                                _tn = (
                                    _mt.get("predicted_n", 0) + _accumulated_predicted_n
                                )
                                _mt["predicted_n"] = _tn
                                _tms = _mt["predicted_ms"]
                                if _tms > 0:
                                    _mt["predicted_per_second"] = _tn / (_tms / 1000.0)
                            yield {
                                "type": "metadata",
                                "usage": {
                                    "prompt_tokens": _fp,
                                    "completion_tokens": _tc,
                                    "total_tokens": _fp + _tc,
                                },
                                "timings": _mt,
                            }
                        return

                    # Safety net caught tool XML -- treat as tool call
                    tool_calls = _safety_tc
                    content_text = _strip_tool_markup(
                        content_accum,
                        final = True,
                    )
                    logger.info(
                        f"Safety net: parsed {len(tool_calls)} tool call(s) "
                        f"from streamed content"
                    )
                else:
                    # ── DRAINING path: assemble tool_calls ──
                    tool_calls = None
                    content_text = content_accum
                    if has_structured_tc:
                        # Filter out incomplete fragments (e.g. from
                        # truncation by max_tokens or disconnect).
                        tool_calls = [
                            tool_calls_acc[i]
                            for i in sorted(tool_calls_acc)
                            if (
                                tool_calls_acc[i]
                                .get("function", {})
                                .get("name", "")
                                .strip()
                            )
                        ] or None
                    if (
                        not tool_calls
                        and auto_heal_tool_calls
                        and any(s in content_accum for s in _TOOL_XML_SIGNALS)
                    ):
                        tool_calls = self._parse_tool_calls_from_text(
                            content_accum,
                        )
                    if tool_calls and not has_structured_tc:
                        content_text = _strip_tool_markup(
                            content_text,
                            final = True,
                        )
                    if tool_calls:
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s) from "
                            f"{'structured delta' if has_structured_tc else 'content text'}"
                        )
                    if not tool_calls:
                        # DRAINING but no tool calls (false positive).
                        # Merge accumulated metrics from prior tool
                        # iterations so they are not silently dropped.
                        yield {"type": "status", "text": ""}
                        if content_accum:
                            # Strip leaked tool-call XML before yielding
                            content_accum = _strip_tool_markup(
                                content_accum, final = True
                            )
                        if content_accum:
                            yield {"type": "content", "text": content_accum}
                        _fu = _iter_usage or {}
                        _fc = _fu.get("completion_tokens", 0)
                        _fp = _fu.get("prompt_tokens", 0)
                        _tc = _fc + _accumulated_completion_tokens
                        if (
                            _iter_usage
                            or _iter_timings
                            or _accumulated_completion_tokens
                        ):
                            _mt = dict(_iter_timings) if _iter_timings else {}
                            if _accumulated_predicted_ms or _accumulated_predicted_n:
                                _mt["predicted_ms"] = (
                                    _mt.get("predicted_ms", 0)
                                    + _accumulated_predicted_ms
                                )
                                _tn = (
                                    _mt.get("predicted_n", 0) + _accumulated_predicted_n
                                )
                                _mt["predicted_n"] = _tn
                                _tms = _mt["predicted_ms"]
                                if _tms > 0:
                                    _mt["predicted_per_second"] = _tn / (_tms / 1000.0)
                            yield {
                                "type": "metadata",
                                "usage": {
                                    "prompt_tokens": _fp,
                                    "completion_tokens": _tc,
                                    "total_tokens": _fp + _tc,
                                },
                                "timings": _mt,
                            }
                        return

                # ── Execute tool calls ──
                _accumulated_completion_tokens += (_iter_usage or {}).get(
                    "completion_tokens", 0
                )
                _it = _iter_timings or {}
                _accumulated_predicted_ms += _it.get("predicted_ms", 0)
                _accumulated_predicted_n += _it.get("predicted_n", 0)

                assistant_msg = {"role": "assistant", "content": content_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                conversation.append(assistant_msg)

                for tc in tool_calls or []:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    raw_args = func.get("arguments", {})

                    if isinstance(raw_args, str):
                        try:
                            arguments = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            if auto_heal_tool_calls:
                                arguments = {"query": raw_args}
                            else:
                                arguments = {"raw": raw_args}
                    else:
                        arguments = raw_args

                    if tool_name == "web_search":
                        _ws_url = (arguments.get("url") or "").strip()
                        if _ws_url:
                            _parsed = urlparse(_ws_url)
                            if _parsed.scheme in ("http", "https") and _parsed.hostname:
                                _ws_host = _parsed.hostname
                                if _ws_host.startswith("www."):
                                    _ws_host = _ws_host[4:]
                                status_text = f"Reading: {_ws_host}"
                            else:
                                status_text = "Reading page..."
                        else:
                            status_text = f"Searching: {arguments.get('query', '')}"
                    elif tool_name == "python":
                        preview = (
                            (arguments.get("code") or "").strip().split("\n")[0][:60]
                        )
                        status_text = (
                            f"Running Python: {preview}"
                            if preview
                            else "Running Python..."
                        )
                    elif tool_name == "terminal":
                        cmd_preview = (arguments.get("command") or "")[:60]
                        status_text = (
                            f"Running: {cmd_preview}"
                            if cmd_preview
                            else "Running command..."
                        )
                    else:
                        status_text = f"Calling: {tool_name}"
                    yield {"type": "status", "text": status_text}

                    yield {
                        "type": "tool_start",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "arguments": arguments,
                    }

                    # ── Duplicate call detection ──────────────
                    # str(dict) is stable here: arguments always comes from
                    # json.loads on the same model output within one request,
                    # so insertion order is deterministic (Python 3.7+).
                    _tc_key = tool_name + str(arguments)
                    _prev = _tool_call_history[-1] if _tool_call_history else None
                    if _prev and _prev[0] == _tc_key and not _prev[1]:
                        result = (
                            "You already made this exact call. "
                            "Do not repeat the same tool call. "
                            "Try a different approach: fetch a URL "
                            "from previous results, use Python to "
                            "process data you already have, or "
                            "provide your final answer now."
                        )
                    else:
                        _effective_timeout = (
                            None if tool_call_timeout >= 9999 else tool_call_timeout
                        )
                        result = execute_tool(
                            tool_name,
                            arguments,
                            cancel_event = cancel_event,
                            timeout = _effective_timeout,
                            session_id = session_id,
                        )

                    yield {
                        "type": "tool_end",
                        "tool_name": tool_name,
                        "tool_call_id": tc.get("id", ""),
                        "result": result,
                    }

                    # Nudge model to try a different approach on errors
                    _error_prefixes = (
                        "Error",
                        "Search failed",
                        "Execution error",
                        "Blocked:",
                        "Exit code",
                        "Failed to fetch",
                        "Failed to resolve",
                        "No query provided",
                    )
                    _is_error = isinstance(result, str) and result.lstrip().startswith(
                        _error_prefixes
                    )
                    _tool_call_history.append((_tc_key, _is_error))
                    # Strip image sentinel before feeding result to the LLM
                    # (the full result with sentinel is still yielded via
                    # tool_end so the frontend can extract image paths).
                    _result_content = result
                    if "\n__IMAGES__:" in _result_content:
                        _result_content = _result_content.rsplit("\n__IMAGES__:", 1)[0]
                    if _is_error:
                        _result_content = (
                            _result_content + "\n\nThe tool call encountered an issue. "
                            "Please try a different approach or rephrase your request."
                        )

                    tool_msg = {
                        "role": "tool",
                        "name": tool_name,
                        "content": _result_content,
                    }
                    tool_call_id = tc.get("id")
                    if tool_call_id:
                        tool_msg["tool_call_id"] = tool_call_id
                    conversation.append(tool_msg)

                # Clear tool status badge before next generation iteration
                yield {"type": "status", "text": ""}
                # Continue the loop to let model respond with context
                continue

            except httpx.ConnectError:
                raise RuntimeError("Lost connection to llama-server")
            except Exception as e:
                if cancel_event is not None and cancel_event.is_set():
                    return
                raise

        # ── Tool iteration cap reached -- synthesize final answer ──
        # The model used all iterations without producing a final text
        # response. Inject a nudge so the final streaming pass produces
        # a useful answer instead of continuing to request tools.
        if max_tool_iterations > 0:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have used all available tool calls. Based on "
                        "everything you have found so far, provide your final "
                        "answer now. Do not call any more tools."
                    ),
                }
            )

        # Clear status
        yield {"type": "status", "text": ""}

        # Final streaming pass with the full conversation context
        stream_payload = {
            "messages": conversation,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            stream_payload["chat_template_kwargs"] = _reasoning_kw
        stream_payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        stream_payload["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
        if stop:
            stream_payload["stop"] = stop
        stream_payload["stream_options"] = {"include_usage": True}

        cumulative = ""
        _last_emitted = ""
        in_thinking = False
        has_content_tokens = False
        reasoning_text = ""
        _metadata_usage = None
        _metadata_timings = None
        _stream_done = False

        try:
            stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
            _auth_headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            )
            with httpx.Client(
                timeout = stream_timeout, limits = httpx.Limits(max_keepalive_connections = 0)
            ) as client:
                with self._stream_with_retry(
                    client,
                    url,
                    stream_payload,
                    cancel_event,
                    headers = _auth_headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode()
                        raise RuntimeError(
                            f"llama-server returned {response.status_code}: {error_body}"
                        )

                    buffer = ""
                    for raw_chunk in self._iter_text_cancellable(
                        response, cancel_event
                    ):
                        buffer += raw_chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue
                            if line == "data: [DONE]":
                                if in_thinking:
                                    if has_content_tokens:
                                        cumulative += "</think>"
                                        yield {
                                            "type": "content",
                                            "text": _strip_tool_markup(
                                                cumulative, final = True
                                            ),
                                        }
                                    else:
                                        cumulative = reasoning_text
                                        yield {"type": "content", "text": cumulative}
                                _stream_done = True
                                break  # exit inner while
                            if not line.startswith("data: "):
                                continue

                            try:
                                chunk_data = json.loads(line[6:])
                                # Capture server timings/usage from final chunks
                                _chunk_timings = chunk_data.get("timings")
                                if _chunk_timings:
                                    _metadata_timings = _chunk_timings
                                _chunk_usage = chunk_data.get("usage")
                                if _chunk_usage:
                                    _metadata_usage = _chunk_usage
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})

                                    reasoning = delta.get("reasoning_content", "")
                                    if reasoning:
                                        reasoning_text += reasoning
                                        if not in_thinking:
                                            cumulative += "<think>"
                                            in_thinking = True
                                        cumulative += reasoning
                                        yield {"type": "content", "text": cumulative}

                                    token = delta.get("content", "")
                                    if token:
                                        has_content_tokens = True
                                        if in_thinking:
                                            cumulative += "</think>"
                                            in_thinking = False
                                        cumulative += token
                                        cleaned = _strip_tool_markup(cumulative)
                                        # Only emit when cleaned text grows (monotonic).
                                        if len(cleaned) > len(_last_emitted):
                                            _last_emitted = cleaned
                                            yield {"type": "content", "text": cleaned}
                            except json.JSONDecodeError:
                                logger.debug(
                                    f"Skipping malformed SSE line: {line[:100]}"
                                )
                        if _stream_done:
                            break  # exit outer for
                    _final_usage = _metadata_usage or {}
                    _final_completion = _final_usage.get("completion_tokens", 0)
                    _final_prompt = _final_usage.get("prompt_tokens", 0)
                    _total_completion = (
                        _final_completion + _accumulated_completion_tokens
                    )
                    if _metadata_usage or _metadata_timings:
                        _merged_timings = (
                            dict(_metadata_timings) if _metadata_timings else {}
                        )
                        if _accumulated_predicted_ms or _accumulated_predicted_n:
                            _merged_timings["predicted_ms"] = (
                                _merged_timings.get("predicted_ms", 0)
                                + _accumulated_predicted_ms
                            )
                            _total_predicted_n = (
                                _merged_timings.get("predicted_n", 0)
                                + _accumulated_predicted_n
                            )
                            _merged_timings["predicted_n"] = _total_predicted_n
                            _total_predicted_ms = _merged_timings["predicted_ms"]
                            if _total_predicted_ms > 0:
                                _merged_timings["predicted_per_second"] = (
                                    _total_predicted_n / (_total_predicted_ms / 1000.0)
                                )
                        yield {
                            "type": "metadata",
                            "usage": {
                                "prompt_tokens": _final_prompt,
                                "completion_tokens": _total_completion,
                                "total_tokens": _final_prompt + _total_completion,
                            },
                            "timings": _merged_timings,
                        }

        except httpx.ConnectError:
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            raise
