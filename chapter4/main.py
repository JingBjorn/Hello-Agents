from chapter4.llmClient import HelloAgentsLLM
from chapter4.ToolExecutor import ToolExecutor
from chapter4.ReAct.ReActAgent import ReActAgent
from chapter4.tools.search import search
from chapter4.llmClient import HelloAgentsLLM
from chapter4.PlanAndSolve.PlanAndSolveAgent import PlanAndSolveAgent
from chapter4.Reflection.ReflectionAgent import ReflectionAgent


def build_tool_executor() -> ToolExecutor:
    executor = ToolExecutor()
    executor.registerTool(
        "Search",
        "一个网页搜索引擎。当需要查询时事、事实或知识库之外的信息时使用。",
        search,
    )
    return executor


def re_act():
    llm_client = HelloAgentsLLM()
    tool_executor = build_tool_executor()

    agent = ReActAgent(
        llm_client=llm_client,
        tool_executor=tool_executor,
    )
    agent.run("华为最新的手机是哪一款？它的主要卖点是什么？")

def plan_and_solve():
    llm_client = HelloAgentsLLM()

    agent = PlanAndSolveAgent(
        llm_client=llm_client,
    )
    agent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")

def reflection():
    llm_client = HelloAgentsLLM()

    agent = ReflectionAgent(
        llm_client=llm_client,
    )
    agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)")


if __name__ == "__main__":
    reflection()