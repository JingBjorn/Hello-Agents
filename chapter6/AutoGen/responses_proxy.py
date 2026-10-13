"""Local, text-only Chat Completions -> Responses adapter.

The upstream is streamed (some Codex gateways require this), but downstream
streaming is buffered until the complete response is available. No tool execution,
image input, or structured-output support is advertised.
"""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from typing import Any
from urllib.parse import urlsplit

import httpx

DEFAULT_MAX_OUTPUT_TOKENS = 32768
DEFAULT_EMPTY_RESPONSE_RETRIES = 1
logger = logging.getLogger(__name__)


class ProxyError(Exception):
    def __init__(self, status: int, message: str, code: str = "proxy_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        for part in content
    ):
        return "\n".join(part["text"] for part in content)
    raise ProxyError(400, "此中转仅支持纯文本消息。", "unsupported_content")


def to_responses_request(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ProxyError(400, "请求必须是 JSON 对象。", "invalid_request")
    allowed = {
        "model", "messages", "stream", "stream_options", "n",
        "temperature", "top_p", "max_tokens", "max_completion_tokens",
        "reasoning_effort", "verbosity",
    }
    unsupported = set(body) - allowed
    if unsupported:
        raise ProxyError(
            400, f"此纯文本中转不支持参数：{', '.join(sorted(unsupported))}",
            "unsupported_parameter",
        )
    if body.get("n", 1) != 1:
        raise ProxyError(400, "仅支持 n=1。", "unsupported_parameter")
    if "stream" in body and not isinstance(body["stream"], bool):
        raise ProxyError(400, "stream 必须是布尔值。", "invalid_request")
    options = body.get("stream_options")
    if options is not None and (
        not isinstance(options, dict)
        or set(options) - {"include_usage"}
        or not isinstance(options.get("include_usage", False), bool)
    ):
        raise ProxyError(400, "不支持该 stream_options。", "unsupported_parameter")
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ProxyError(400, "缺少 model。", "invalid_request")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ProxyError(400, "messages 必须是非空数组。", "invalid_request")

    instructions = []
    inputs = []
    for message in messages:
        if not isinstance(message, dict):
            raise ProxyError(400, "消息必须是对象。", "invalid_request")
        if set(message) - {"role", "content", "name"}:
            raise ProxyError(400, "不支持工具调用或其他消息字段。", "unsupported_content")
        role = message.get("role")
        text = _text(message.get("content"))
        if role in ("system", "developer"):
            instructions.append(text)
        elif role in ("user", "assistant"):
            if message.get("name"):
                text = f"[{message['name']}]\n{text}"
            inputs.append({
                "role": role,
                "content": [{
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": text,
                }],
            })
        else:
            raise ProxyError(400, "仅支持 system/developer/user/assistant 消息。",
                             "unsupported_content")
    if not inputs:
        raise ProxyError(400, "至少需要一条 user 或 assistant 消息。", "invalid_request")

    result = {
        "model": body["model"],
        "instructions": "\n\n".join(instructions),
        "input": inputs,
        "store": False,
        "stream": True,
    }
    for key in ("temperature", "top_p"):
        if key in body:
            result[key] = body[key]
    maximum = body.get("max_completion_tokens", body.get("max_tokens"))
    if maximum is not None:
        result["max_output_tokens"] = maximum
    if "reasoning_effort" in body:
        result["reasoning"] = {"effort": body["reasoning_effort"]}
    if "verbosity" in body:
        result["text"] = {"verbosity": body["verbosity"]}
    return result


def _sse_events(lines):
    """Parse SSE data, including multiline events and a missing trailing blank."""
    data = []
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


class _StreamOutput:
    """Retain public output when a gateway omits it from its terminal event."""

    def __init__(self):
        self.items = {}
        self.parts = {}

    def add(self, event):
        kind = event.get("type", "")
        if kind == "response.output_item.done":
            self.items[event["output_index"]] = event["item"]
        elif kind in (
            "response.output_text.delta", "response.output_text.done",
            "response.refusal.delta", "response.refusal.done",
            "response.content_part.done",
        ):
            key = (event["output_index"], event["content_index"])
            if kind == "response.content_part.done":
                self.parts[key] = event["part"]
                return
            refusal = kind.startswith("response.refusal.")
            field = "refusal" if refusal else "text"
            part_type = "refusal" if refusal else "output_text"
            if kind.endswith(".done"):
                self.parts[key] = {"type": part_type, field: event[field]}
            else:
                part = self.parts.setdefault(key, {"type": part_type, field: ""})
                part[field] += event["delta"]

    def merge(self, response):
        # Terminal output is authoritative. Fill only missing/empty content;
        # never append deltas to the full text (which would duplicate it).
        items = dict(self.items)
        for index, item in enumerate(response.get("output") or []):
            if item.get("content") or index not in items:
                items[index] = item
        for (index, content_index), part in sorted(self.parts.items()):
            item = items.setdefault(
                index, {"type": "message", "role": "assistant", "content": []}
            )
            if item.get("type") != "message":
                continue
            content = list(item.get("content") or [])
            if content_index == len(content):
                content.append(part)
            elif content_index < len(content):
                existing = content[content_index]
                field = "refusal" if existing.get("type") == "refusal" else "text"
                if not existing.get(field):
                    content[content_index] = part
            items[index] = {**item, "content": content}
        return {**response, "output": [items[index] for index in sorted(items)]}


def read_response(response: httpx.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        response.read()
        try:
            error = response.json().get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            message = None
        raise ProxyError(response.status_code, message or "上游返回非 JSON 错误。",
                         "upstream_error")
    try:
        # Some gateways label an SSE body as text/plain. Inspect the first
        # nonempty line instead of trusting Content-Type, without buffering SSE.
        lines = iter(response.iter_lines())
        first = next((line.lstrip("\ufeff") for line in lines if line.strip()), "")
        if not first.startswith(("data:", "event:", ":", "id:", "retry:")):
            result = json.loads("\n".join(chain([first], lines)))
        else:
            result = None
            output = _StreamOutput()
            for data in _sse_events(chain([first], lines)):
                if data == "[DONE]":
                    break
                event = json.loads(data)
                kind = event.get("type")
                if kind in ("response.completed", "response.incomplete"):
                    result = output.merge(event["response"])
                    # Terminal event: do not wait for a gateway to close its socket.
                    break
                if kind in ("error", "response.failed"):
                    error = event.get("error") or event.get("response", {}).get("error") or event
                    message = error.get("message", "上游 Responses 生成失败。")
                    raise ProxyError(502, message, "upstream_error")
                output.add(event)
            if result is None:
                raise ProxyError(502, "上游流提前结束，缺少 Responses 完成事件。",
                                 "incomplete_stream")
        if not isinstance(result, dict):
            raise ValueError("Expected an object")
        if result.get("error") or result.get("status") in ("failed", "cancelled"):
            error = result.get("error") or {}
            raise ProxyError(502, error.get("message", "上游生成失败。"), "upstream_error")
        return result
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ProxyError(502, "上游返回了无法解析的 Responses 数据。",
                         "invalid_upstream_response") from exc


def to_chat_completion(response: dict[str, Any], model: str) -> dict[str, Any]:
    texts = []
    refusals = []
    for item in response.get("output", []):
        if item.get("type") == "reasoning":
            continue  # Never expose internal reasoning as assistant text.
        if item.get("type") != "message":
            raise ProxyError(502, "上游返回了此纯文本中转不支持的输出类型。",
                             "unsupported_output")
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content["text"])
            elif content.get("type") == "refusal":
                refusals.append(content["refusal"])
            else:
                raise ProxyError(502, "上游返回非文本消息内容。", "unsupported_output")
    if not any(text.strip() for text in texts + refusals):
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason == "max_output_tokens":
            detail = "上游输出预算已耗尽，但没有生成可用文本。"
        else:
            detail = "上游没有返回可用文本（无法仅凭空输出确定原因）。"
        raise ProxyError(
            502, detail + " 可提高 AUTOGEN_MAX_OUTPUT_TOKENS，或在模型支持时"
            "设置 AUTOGEN_REASONING_EFFORT=low；若仍失败，请检查网关响应。",
            "empty_response",
        )
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("input_tokens") or 0
    completion_tokens = usage.get("output_tokens") or 0
    reason = (response.get("incomplete_details") or {}).get("reason")
    finish = "length" if reason == "max_output_tokens" else "stop"
    if reason == "content_filter":
        finish = "content_filter"
    message = {"role": "assistant", "content": "".join(texts) or "".join(refusals)}
    if refusals:
        message["refusal"] = "".join(refusals)
    return {
        "id": f"chatcmpl-{response.get('id') or secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(response.get("created_at") or time.time()),
        # Echo the requested alias so AutoGen does not misidentify gateway models.
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens") or prompt_tokens + completion_tokens,
        },
    }


def chat_chunks(completion, include_usage=False):
    """Valid buffered Chat Completions SSE, not token-by-token live streaming."""
    base = {key: completion[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    choice = completion["choices"][0]
    for delta, finish in (
        ({"role": "assistant"}, None),
        ({"content": choice["message"]["content"]}, None),
        ({}, choice["finish_reason"]),
    ):
        yield {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if include_usage:
        yield {**base, "choices": [], "usage": completion["usage"]}


class ResponsesProxy:
    """Ephemeral loopback HTTP server, usable as a synchronous context manager."""

    def __init__(self, upstream_url: str, upstream_key: str, *, timeout=180.0,
                 max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                 empty_response_retries=DEFAULT_EMPTY_RESPONSE_RETRIES,
                 reasoning_effort=None,
                 transport: httpx.BaseTransport | None = None):
        parsed = urlsplit(upstream_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("OPENAI_BASE_URL 必须是有效的 HTTP(S) API 根地址。")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OPENAI_BASE_URL 不应包含认证信息、查询参数或片段。")
        if not upstream_key:
            raise ValueError("请在 .env 中设置 OPENAI_API_KEY。")
        if timeout <= 0:
            raise ValueError("AUTOGEN_PROXY_TIMEOUT 必须大于 0。")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("AUTOGEN_MAX_OUTPUT_TOKENS 必须是正整数。")
        if type(empty_response_retries) is not int or not 0 <= empty_response_retries <= 3:
            raise ValueError("AUTOGEN_EMPTY_RESPONSE_RETRIES 必须是 0 到 3 的整数。")
        self._max_output_tokens = max_output_tokens
        self._empty_response_retries = empty_response_retries
        self._reasoning_effort = reasoning_effort
        self.api_key = secrets.token_urlsafe(32)
        self._upstream_key = upstream_key
        self._url = upstream_url.rstrip("/") + "/responses"
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=15),
            transport=transport,
            follow_redirects=False,
        )
        self._server = None
        self._thread = None

    @property
    def base_url(self):
        if self._server is None:
            raise RuntimeError("中转尚未启动。")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def complete(self, body):
        payload = to_responses_request(body)
        payload.setdefault("max_output_tokens", self._max_output_tokens)
        if self._reasoning_effort:
            payload.setdefault("reasoning", {"effort": self._reasoning_effort})
        previous_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            for attempt in range(self._empty_response_retries + 1):
                with self._http.stream(
                    "POST", self._url,
                    headers={"Authorization": f"Bearer {self._upstream_key}",
                             "Accept": "text/event-stream"},
                    json=payload,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise ProxyError(502, "上游返回重定向，请确认 API 根地址。",
                                         "upstream_redirect")
                    result = read_response(response)
                try:
                    completion = to_chat_completion(result, body["model"])
                except ProxyError as exc:
                    reason = (result.get("incomplete_details") or {}).get("reason")
                    if (exc.code != "empty_response"
                            or reason == "content_filter"
                            or attempt == self._empty_response_retries):
                        raise
                    usage = result.get("usage") or {}
                    prompt = usage.get("input_tokens") or 0
                    output = usage.get("output_tokens") or 0
                    previous_usage["prompt_tokens"] += prompt
                    previous_usage["completion_tokens"] += output
                    previous_usage["total_tokens"] += usage.get("total_tokens") or prompt + output
                    logger.warning(
                        "上游返回空文本，正在重试 (%d/%d)，可能产生额外 API 费用。",
                        attempt + 1, self._empty_response_retries,
                    )
                    continue
                for key, value in previous_usage.items():
                    completion["usage"][key] += value
                return completion
        except httpx.TimeoutException as exc:
            raise ProxyError(504, "上游请求超时，可调整 AUTOGEN_PROXY_TIMEOUT。",
                             "upstream_timeout") from exc
        except httpx.HTTPError as exc:
            raise ProxyError(502, "无法连接上游 Responses 接口。", "upstream_connection") from exc

    def __enter__(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Do not log prompts, credentials, or upstream responses.

            def send_json(self, status, value):
                data = json.dumps(value, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                try:
                    if self.path != "/v1/chat/completions":
                        raise ProxyError(404, "仅支持 /v1/chat/completions。", "not_found")
                    supplied = self.headers.get("Authorization", "").encode("utf-8")
                    expected = f"Bearer {proxy.api_key}".encode("utf-8")
                    if not hmac.compare_digest(supplied, expected):
                        raise ProxyError(401, "本地中转密钥无效。", "invalid_api_key")
                    if self.headers.get("Transfer-Encoding"):
                        raise ProxyError(400, "不支持分块请求体。", "invalid_request")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4 * 1024 * 1024:
                        raise ProxyError(413, "请求体为空或超过 4 MiB。", "invalid_request")
                    self.connection.settimeout(30)
                    body = json.loads(self.rfile.read(length))
                    completion = proxy.complete(body)
                    if not body.get("stream"):
                        self.send_json(200, completion)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    options = body.get("stream_options") or {}
                    for chunk in chat_chunks(completion, options.get("include_usage", False)):
                        self.wfile.write(
                            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                        )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except ProxyError as exc:
                    # Some gateways echo their request in errors: redact the real key.
                    self.send_json(exc.status, {"error": {
                        "message": str(exc).replace(proxy._upstream_key, "[REDACTED]"),
                        "type": "proxy_error", "code": exc.code,
                    }})
                except (ValueError, UnicodeError):
                    self.send_json(400, {"error": {"message": "无效的 JSON 请求或请求头。",
                                                  "code": "invalid_request"}})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    self.send_json(502, {"error": {"message": "中转无法处理上游响应。",
                                                  "code": "proxy_error"}})

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except Exception:
            self._http.close()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
            self._server = None
        self._http.close()
