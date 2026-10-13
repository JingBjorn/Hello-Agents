"""Offline unit and HTTP integration tests; no real model calls."""

import asyncio
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
from autogen_agentchat.agents import UserProxyAgent
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage
from openai import OpenAI

from client import configured_model_client, create_openai_model_client, run_team
from productManager import create_product_manager
from engineer import create_engineer
from codeReviewer import create_code_reviewer
from userProxy import create_user_proxy
from responses_proxy import (
    DEFAULT_MAX_OUTPUT_TOKENS, ProxyError, ResponsesProxy, read_response, to_chat_completion,
    to_responses_request,
)


def response_fixture(**overrides):
    return {
        "id": "resp_test",
        "status": "completed",
        "created_at": 1720000000,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "你好，OK"}]},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        **overrides,
    }


def sse_fixture(response=None):
    event = {"type": "response.completed", "response": response or response_fixture()}
    return f": heartbeat\n\nevent: response.completed\ndata: {json.dumps(event)}\n\n"


def event_stream(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


class MappingTests(unittest.TestCase):
    def test_stream_text_retained_when_terminal_output_is_empty(self):
        events = [
            {"type": "response.output_text.delta", "output_index": 1,
             "content_index": 0, "delta": "你好"},
            {"type": "response.output_text.delta", "output_index": 1,
             "content_index": 0, "delta": "，OK"},
            {"type": "response.completed", "response": response_fixture(output=[])},
        ]
        result = read_response(httpx.Response(200, text=event_stream(*events)))
        self.assertEqual(to_chat_completion(result, "alias")["choices"][0]["message"]["content"],
                         "你好，OK")

    def test_stream_done_events_and_terminal_do_not_duplicate_text(self):
        for terminal_output in ([], response_fixture()["output"]):
            for kind, fields in (
                ("response.output_text.done", {"content_index": 0, "text": "你好，OK"}),
                ("response.content_part.done", {"content_index": 0, "part":
                    {"type": "output_text", "text": "你好，OK"}}),
                ("response.output_item.done", {"item": response_fixture()["output"][1]}),
            ):
                with self.subTest(kind=kind, terminal_output=terminal_output):
                    text = event_stream(
                        {"type": "response.output_text.delta", "output_index": 1,
                         "content_index": 0, "delta": "你好"},
                        {"type": kind, "output_index": 1, **fields},
                        {"type": "response.completed",
                         "response": response_fixture(output=terminal_output)},
                    )
                    result = to_chat_completion(read_response(httpx.Response(200, text=text)), "alias")
                    self.assertEqual(result["choices"][0]["message"]["content"], "你好，OK")

    def test_stream_refusal_retained_and_reasoning_never_exposed(self):
        text = event_stream(
            {"type": "response.reasoning_text.delta", "delta": "private reasoning"},
            {"type": "response.refusal.delta", "output_index": 0,
             "content_index": 0, "delta": "不能"},
            {"type": "response.refusal.done", "output_index": 0,
             "content_index": 0, "refusal": "不能帮助"},
            {"type": "response.completed", "response": response_fixture(output=[])},
        )
        result = to_chat_completion(read_response(httpx.Response(200, text=text)), "alias")
        self.assertEqual(result["choices"][0]["message"]["refusal"], "不能帮助")
        self.assertNotIn("private reasoning", json.dumps(result))

    def test_stream_text_without_terminal_event_still_fails(self):
        text = event_stream({"type": "response.output_text.delta", "output_index": 0,
                             "content_index": 0, "delta": "partial"})
        with self.assertRaises(ProxyError) as error:
            read_response(httpx.Response(200, text=text))
        self.assertEqual(error.exception.code, "incomplete_stream")

    def test_blank_text_is_empty_and_budget_diagnostic_is_specific(self):
        for text in ("", " \n "):
            with self.subTest(text=text), self.assertRaises(ProxyError) as error:
                to_chat_completion(response_fixture(
                    status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                    output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}],
                ), "alias")
            self.assertEqual(error.exception.code, "empty_response")
            self.assertIn("输出预算已耗尽", str(error.exception))

    def test_messages_and_parameters(self):
        payload = to_responses_request({
            "model": "custom-model",
            "messages": [
                {"role": "system", "content": "系统"},
                {"role": "developer", "content": "开发者"},
                {"role": "user", "name": "产品经理",
                 "content": [{"type": "text", "text": "需求"}]},
                {"role": "assistant", "content": "方案"},
                {"role": "user", "content": "继续"},
            ],
            "max_completion_tokens": 100,
            "reasoning_effort": "low", "verbosity": "low",
            "temperature": 0.2, "top_p": 0.9,
        })
        self.assertEqual(payload["instructions"], "系统\n\n开发者")
        self.assertEqual(payload["input"][0]["content"][0]["text"], "[产品经理]\n需求")
        self.assertEqual(payload["input"][1]["content"][0]["type"], "output_text")
        self.assertEqual(len(payload["input"]), 3)
        self.assertEqual(payload["max_output_tokens"], 100)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])

    def test_reject_unsupported_and_invalid_inputs(self):
        base = {"model": "custom", "messages": [{"role": "user", "content": "hi"}]}
        for extra in (
            {"tools": []}, {"response_format": {"type": "json_object"}}, {"n": 2},
            {"stream": "true"}, {"messages": []}, {"stream_options": {"bad": True}},
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            {"messages": [{"role": "tool", "content": "result"}]},
            {"messages": [{"role": "assistant", "content": "", "tool_calls": []}]},
        ):
            with self.subTest(extra=extra), self.assertRaises(ProxyError) as error:
                to_responses_request({**base, **extra})
            self.assertEqual(error.exception.status, 400)

    def test_sse_and_json(self):
        for response in (
            httpx.Response(200, json=response_fixture()),
            httpx.Response(200, text=sse_fixture(),
                           headers={"content-type": "text/event-stream"}),
        ):
            self.assertEqual(read_response(response)["id"], "resp_test")

    def test_sse_with_gateway_text_plain_content_type(self):
        for headers in ({}, {"content-type": "text/plain; charset=utf-8"}):
            response = httpx.Response(200, text=sse_fixture(), headers=headers)
            self.assertEqual(read_response(response)["id"], "resp_test")

    def test_sse_errors_are_not_success(self):
        for text in (
            "data: [DONE]\n\n",
            'data: {"type":"error","message":"failed"}\n\n',
            'data: {"type":"response.failed","response":{"error":{"message":"failed"}}}\n\n',
            "data: not-json\n\n",
        ):
            with self.subTest(text=text), self.assertRaises(ProxyError):
                read_response(httpx.Response(200, text=text,
                                             headers={"content-type": "text/event-stream"}))

    def test_error_status_preserved(self):
        with self.assertRaises(ProxyError) as error:
            read_response(httpx.Response(429, json={"error": {"message": "quota"}}))
        self.assertEqual(error.exception.status, 429)

    def test_output_and_usage(self):
        result = to_chat_completion(response_fixture(), "alias")
        self.assertEqual(result["model"], "alias")
        self.assertEqual(result["choices"][0]["message"]["content"], "你好，OK")
        self.assertEqual(result["usage"]["prompt_tokens"], 12)
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        result = to_chat_completion(response_fixture(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        ), "alias")
        self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_refusal_and_empty_output(self):
        result = to_chat_completion(response_fixture(output=[
            {"type": "message", "content": [{"type": "refusal", "refusal": "不能帮助"}]}
        ]), "alias")
        self.assertEqual(result["choices"][0]["message"]["refusal"], "不能帮助")
        with self.assertRaises(ProxyError):
            to_chat_completion(response_fixture(output=[]), "alias")
        with self.assertRaises(ProxyError):
            to_chat_completion(response_fixture(output=[{"type": "function_call"}]), "alias")


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.mock_response = lambda: httpx.Response(
            200, text=sse_fixture(), headers={"content-type": "text/event-stream"}
        )

        def upstream(request):
            self.requests.append(request)
            return self.mock_response()

        self.proxy = ResponsesProxy(
            "https://gateway.example/custom/v1", "test-upstream-secret",
            transport=httpx.MockTransport(upstream),
        )
        self.proxy.__enter__()
        self.addCleanup(self.proxy.__exit__, None, None, None)
        self.client = OpenAI(
            base_url=self.proxy.base_url, api_key=self.proxy.api_key,
            max_retries=0, http_client=httpx.Client(trust_env=False),
        )
        self.addCleanup(self.client.close)

    def test_real_openai_client_nonstream(self):
        result = self.client.chat.completions.create(
            model="custom", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertEqual(result.choices[0].message.content, "你好，OK")
        request = self.requests[0]
        self.assertEqual(str(request.url), "https://gateway.example/custom/v1/responses")
        self.assertEqual(request.headers["authorization"], "Bearer test-upstream-secret")
        self.assertNotIn(self.proxy.api_key, request.content.decode())
        self.assertEqual(json.loads(request.content)["max_output_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)

    def test_real_openai_client_stream(self):
        chunks = list(self.client.chat.completions.create(
            model="custom", messages=[{"role": "user", "content": "hi"}],
            stream=True, stream_options={"include_usage": True},
        ))
        text = "".join(chunk.choices[0].delta.content or ""
                       for chunk in chunks if chunk.choices)
        self.assertEqual(text, "你好，OK")
        self.assertEqual(chunks[-1].usage.total_tokens, 16)
        self.assertEqual(chunks[-2].choices[0].finish_reason, "stop")

    def test_authentication_and_invalid_body(self):
        with httpx.Client(trust_env=False) as client:
            url = self.proxy.base_url + "/chat/completions"
            self.assertEqual(client.post(url, json={}).status_code, 401)
            response = client.post(url, content="invalid-json",
                                   headers={"Authorization": f"Bearer {self.proxy.api_key}"})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.requests, [])

    def test_errors_and_secret_redaction(self):
        self.mock_response = lambda: httpx.Response(403, json={
            "error": {"message": "bad key test-upstream-secret"}
        })
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                self.proxy.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self.proxy.api_key}"},
                json={"model": "custom", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("test-upstream-secret", response.text)

    def test_actual_autogen_client_with_agent_names_and_chinese_history(self):
        async def check():
            client = create_openai_model_client(
                base_url=self.proxy.base_url, api_key=self.proxy.api_key
            )
            try:
                manager = create_product_manager(client)
                engineer = create_engineer(client)
                reviewer = create_code_reviewer(client)
                self.assertEqual(create_user_proxy().name, "user_proxy")
                result = await client.create([
                    SystemMessage(content="你是工程师"),
                    UserMessage(content="写应用", source=manager.name),
                    AssistantMessage(content="已有方案", source=engineer.name),
                    UserMessage(content="请继续", source=reviewer.name),
                ])
                self.assertEqual(result.content, "你好，OK")
                self.assertEqual(result.usage.prompt_tokens, 12)
            finally:
                await client.close()

        asyncio.run(check())
        sent = json.loads(self.requests[0].content)
        self.assertEqual(len(sent["input"]), 3)
        self.assertIn("product_manager", sent["input"][0]["content"][0]["text"])
        self.assertIn("code_reviewer", sent["input"][2]["content"][0]["text"])

    def test_empty_response_retries_with_same_payload_and_accounts_for_usage(self):
        self.mock_response = lambda: httpx.Response(200, json=response_fixture(
            output=[] if len(self.requests) == 1 else response_fixture()["output"],
        ))
        result = self.client.chat.completions.create(
            model="custom", messages=[{"role": "user", "content": "可以"}],
            max_completion_tokens=40000, reasoning_effort="low",
        )
        self.assertEqual(result.choices[0].message.content, "你好，OK")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0].content, self.requests[1].content)
        self.assertEqual(json.loads(self.requests[0].content)["max_output_tokens"], 40000)
        self.assertEqual(result.usage.total_tokens, 32)

    def test_permanent_empty_response_stops_at_retry_limit(self):
        self.mock_response = lambda: httpx.Response(200, json=response_fixture(output=[]))
        with httpx.Client(trust_env=False) as client:
            response = client.post(
                self.proxy.base_url + "/chat/completions",
                headers={"Authorization": f"Bearer {self.proxy.api_key}"},
                json={"model": "custom", "messages": [{"role": "user", "content": "hi"}]},
            )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "empty_response")
        self.assertIn("AUTOGEN_MAX_OUTPUT_TOKENS", response.json()["error"]["message"])

    def test_team_continues_after_user_feedback_and_empty_response(self):
        def upstream():
            if len(self.requests) == 4:
                return httpx.Response(200, json=response_fixture(
                    status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                    output=[{"type": "reasoning", "summary": []}],
                ))
            text = "TERMINATE" if len(self.requests) == 5 else "请继续"
            return httpx.Response(200, text=sse_fixture(response_fixture(output=[
                {"type": "message", "content": [{"type": "output_text", "text": text}]}
            ])))

        self.mock_response = upstream

        async def input_func(prompt, cancellation_token=None):
            return "可以"

        async def check():
            client = create_openai_model_client(base_url=self.proxy.base_url, api_key=self.proxy.api_key)
            try:
                with patch("client.create_user_proxy", return_value=UserProxyAgent(
                    name="user_proxy", input_func=input_func
                )), contextlib.redirect_stdout(io.StringIO()):
                    result = await run_team(client)
                self.assertIn("TERMINATE", result.stop_reason)
                self.assertTrue(any(msg.source == "user_proxy" and msg.content == "可以"
                                    for msg in result.messages))
            finally:
                await client.close()

        asyncio.run(check())
        self.assertEqual(len(self.requests), 5)
        self.assertEqual(self.requests[3].content, self.requests[4].content)


class RecoveryConfigurationTests(unittest.TestCase):
    def test_retry_can_be_disabled_and_nonempty_errors_are_not_retried(self):
        for upstream, code, retries in (
            (httpx.Response(200, json=response_fixture(output=[])), "empty_response", 0),
            (httpx.Response(401, json={"error": {"message": "auth"}}), "upstream_error", 1),
            (httpx.Response(429, json={"error": {"message": "quota"}}), "upstream_error", 1),
            (httpx.Response(200, text="data: [DONE]\n\n"), "incomplete_stream", 1),
            (httpx.Response(200, json=response_fixture(output=[{"type": "function_call"}])),
             "unsupported_output", 1),
            (httpx.Response(200, json=response_fixture(
                status="incomplete", incomplete_details={"reason": "content_filter"}, output=[],
            )), "empty_response", 1),
        ):
            requests = []

            def handler(request):
                requests.append(request)
                return upstream

            with self.subTest(code=code, retries=retries):
                proxy = ResponsesProxy("https://gateway.example/v1", "secret",
                                       empty_response_retries=retries,
                                       transport=httpx.MockTransport(handler))
                try:
                    with self.assertRaises(ProxyError) as error:
                        proxy.complete({"model": "custom", "messages": [{"role": "user", "content": "hi"}]})
                    self.assertEqual(error.exception.code, code)
                    self.assertEqual(len(requests), 1)
                finally:
                    proxy.__exit__()

    def test_invalid_configuration(self):
        for config in (
            {"max_output_tokens": 0}, {"max_output_tokens": -1},
            {"empty_response_retries": -1}, {"empty_response_retries": 4},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ResponsesProxy("https://gateway.example/v1", "secret", **config)

    def test_environment_configuration_and_downstream_timeout(self):
        async def check():
            with patch.dict("os.environ", {
                "AUTOGEN_API_MODE": "responses", "AUTOGEN_PROXY_TIMEOUT": "30",
                "AUTOGEN_MAX_OUTPUT_TOKENS": "40000", "AUTOGEN_EMPTY_RESPONSE_RETRIES": "2",
                "AUTOGEN_REASONING_EFFORT": "low",
            }), patch("client.ResponsesProxy") as proxy, patch("client.create_openai_model_client") as factory:
                async def close():
                    pass
                factory.return_value.close = close
                async with configured_model_client():
                    pass
                self.assertEqual(proxy.call_args.kwargs["max_output_tokens"], 40000)
                self.assertEqual(proxy.call_args.kwargs["reasoning_effort"], "low")
                self.assertEqual(proxy.call_args.kwargs["empty_response_retries"], 2)
                self.assertEqual(factory.call_args.kwargs["timeout"], 105)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
