import os
import asyncio
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).resolve().parent / "../../.env")

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from productManager import create_product_manager
from engineer import create_engineer
from codeReviewer import create_code_reviewer
from userProxy import create_user_proxy
from responses_proxy import (
    DEFAULT_EMPTY_RESPONSE_RETRIES, DEFAULT_MAX_OUTPUT_TOKENS, ResponsesProxy,
)

USER_TASK = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""

def create_openai_model_client(*, base_url=None, api_key=None, timeout=None):
    """创建并配置 OpenAI 模型客户端"""
    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        # Responses 不接收 Chat Completions 的 name 字段，用文本前缀保留角色。
        include_name_in_message=False,
        add_name_prefixes=True,
        timeout=timeout if timeout is not None else float(os.getenv("AUTOGEN_PROXY_TIMEOUT", "180")) + 15,
        max_retries=0,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )


@asynccontextmanager
async def configured_model_client():
    """默认自动启动 Responses 中转；退出时释放 HTTP 客户端和监听端口。"""
    mode = os.getenv("AUTOGEN_API_MODE", "responses")
    if mode not in ("responses", "chat_completions"):
        raise ValueError("AUTOGEN_API_MODE 只能是 responses 或 chat_completions。")
    if mode == "chat_completions":
        client = create_openai_model_client()
        try:
            yield client
        finally:
            await client.close()
        return
    timeout = float(os.getenv("AUTOGEN_PROXY_TIMEOUT", "180"))
    retries = int(os.getenv("AUTOGEN_EMPTY_RESPONSE_RETRIES", str(DEFAULT_EMPTY_RESPONSE_RETRIES)))
    with ResponsesProxy(
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("OPENAI_API_KEY", ""),
        timeout=timeout,
        max_output_tokens=int(os.getenv("AUTOGEN_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
        empty_response_retries=retries,
        reasoning_effort=os.getenv("AUTOGEN_REASONING_EFFORT") or None,
    ) as proxy:
        print(f"🔀 本地中转已启动：{proxy.base_url} → 上游 /responses")
        client = create_openai_model_client(
            base_url=proxy.base_url, api_key=proxy.api_key,
            timeout=timeout * (retries + 1) + 15,
        )
        try:
            yield client
        finally:
            await client.close()


async def check_connection():
    """仅发送一条短消息，不运行团队或执行模型生成的代码。"""
    from autogen_core.models import UserMessage

    async with configured_model_client() as client:
        result = await client.create([
            UserMessage(content="请只回复 OK。", source="connection_check")
        ])
        print(f"✅ 接口连通，模型回复：{result.content}")


async def run_software_development_team():
    """运行软件开发团队协作"""
    
    print("🔧 正在初始化模型客户端...")
    
    async with configured_model_client() as model_client:
        return await run_team(model_client)


async def run_team(model_client):
    print("👥 正在创建智能体团队...")
    
    # 创建智能体团队
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()

    # 添加终止条件
    termination = TextMentionTermination("TERMINATE")

    # 创建团队聊天
    team_chat = RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer, 
            code_reviewer,
            user_proxy
        ],
        termination_condition=termination,
        max_turns=20,  # 增加最大轮次
    )

    # 定义开发任务
    task = USER_TASK

    # 执行团队协作
    print("🚀 启动 AutoGen 软件开发团队协作...")
    print("=" * 60)
    
    # 使用 Console 来显示对话过程
    result = await Console(team_chat.run_stream(task=task))

    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")
    
    return result

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGen 软件开发团队（自动 Responses 中转）")
    parser.add_argument("--check", action="store_true", help="仅检查模型连通性，会调用 API，空响应可能重试")
    args = parser.parse_args()
    try:
        if args.check:
            asyncio.run(check_connection())
            raise SystemExit(0)
        # 运行异步协作流程
        result = asyncio.run(run_software_development_team())
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
        
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env、模型能力参数及 AUTOGEN_API_MODE。")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
