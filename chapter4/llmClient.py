import json
import os
import urllib.error
import urllib.request
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv(Path(__file__).resolve().parent / "../.env")

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    支持 OpenAI 兼容的流式 Chat 接口，以及本地 Runtime 中转的非流式 Responses 接口。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None, backend: str = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.backend = (backend or os.getenv("LLM_BACKEND", "openai")).strip().lower()
        if self.backend not in {"openai", "local_runtime"}:
            raise ValueError("LLM_BACKEND 必须是 openai 或 local_runtime。")
        self.timeout = timeout if timeout is not None else int(os.getenv("LLM_TIMEOUT", "60"))

        if self.backend == "local_runtime":
            self.model = model or os.getenv("LOCAL_RUNTIME_MODEL_NAME")
            if not self.model:
                raise ValueError("请通过 model 参数或 LOCAL_RUNTIME_MODEL_NAME 提供本地 Runtime 的模型 ID。")
            self.base_url = (baseUrl or os.getenv("LOCAL_RUNTIME_BASE_URL", "http://127.0.0.1:38471")).rstrip("/")
            # 同时接受根地址和以 /v1 结尾的地址，避免重复拼接。
            if self.base_url.endswith("/v1"):
                self.base_url = self.base_url[:-3]
            self.max_output_tokens = int(os.getenv("LOCAL_RUNTIME_MAX_OUTPUT_TOKENS", "2048"))
            if self.max_output_tokens <= 0:
                raise ValueError("LOCAL_RUNTIME_MAX_OUTPUT_TOKENS 必须为正整数。")
            # 本地 Runtime 负责认证，不向它发送其他供应商的 API Key。
            self.client = None
        else:
            self.model = model or os.getenv("OPENAI_MODEL_NAME")
            apiKey = apiKey or os.getenv("OPENAI_API_KEY")
            baseUrl = baseUrl or os.getenv("OPENAI_BASE_URL")

            if not all([self.model, apiKey, baseUrl]):
                raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

            self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=self.timeout)

    def _think_local_runtime(self, messages: List[Dict[str, str]]) -> str:
        """通过本地 Runtime 发送字符串 input，不假设支持消息数组或流式输出。"""
        if not messages or any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("本地 Runtime 模式需要非空的纯文本消息列表。")
        # 单条用户消息直接传入；多条消息保留角色标签并拼成文本。
        # 这是文本兼容方式，不等同于 API 原生的 system/user 角色隔离。
        if len(messages) == 1 and messages[0].get("role") == "user":
            prompt = messages[0]["content"]
        else:
            prompt = "\n\n".join(
                f"[{message.get('role', 'user')}]\n{message['content']}"
                for message in messages
            )
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        output_text = result.get("output_text")
        if not isinstance(output_text, str):
            raise ValueError("本地 Runtime 响应缺少字符串 output_text，请检查接口返回格式。")
        print("✅ 大语言模型响应成功:")
        print(output_text)
        return output_text
    
    def _think_openai(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """通过 OpenAI API 发送字符串 input，假设支持消息数组和流式输出。"""
        if not messages or any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("OpenAI API 模式需要非空的纯文本消息列表。")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        # 处理流式响应
        print("✅ 大语言模型响应成功:")
        collected_content = []
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            collected_content.append(content)
        print()  # 在流式输出结束后换行
        return "".join(collected_content)
    
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            if self.backend == "local_runtime":
                # 此模式仅发送基础请求字段，不额外传递 temperature。
                return self._think_local_runtime(messages)
            else:
                return self._think_openai(messages, temperature)
        except urllib.error.HTTPError as e:
            print(f"❌ 本地 Runtime 返回 HTTP {e.code}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            print(f"❌ 无法连接本地 Runtime: {e.reason}。请确认本机 Runtime 已启动。")
            return None
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None
    
# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)
    except ValueError as e:
        print(f"❌ 错误: {e}")
