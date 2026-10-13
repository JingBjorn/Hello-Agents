from autogen_agentchat.agents import UserProxyAgent

USER_PROXY_DESCRIPTION = """用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。"""

def create_user_proxy():
    """创建用户代理智能体"""
    return UserProxyAgent(
        name="user_proxy",
        description=USER_PROXY_DESCRIPTION,
    )
