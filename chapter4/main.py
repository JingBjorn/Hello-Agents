from chapter4.llmClient import HelloAgentsLLM
from chapter4.ToolExecutor import ToolExecutor
from chapter4.ReAct.ReActAgent import ReActAgent
from chapter4.tools.search import search


def build_tool_executor() -> ToolExecutor:
    executor = ToolExecutor()
    executor.registerTool(
        "Search",
        "一个网页搜索引擎。当需要查询时事、事实或知识库之外的信息时使用。",
        search,
    )
    return executor


def main():
    llm_client = HelloAgentsLLM()
    tool_executor = build_tool_executor()

    agent = ReActAgent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )
    agent.run("华为最新的手机是哪一款？它的主要卖点是什么？")


if __name__ == "__main__":
    main()