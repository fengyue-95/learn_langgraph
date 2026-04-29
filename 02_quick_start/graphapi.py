# ==========================================
# LangGraph Graph API 完整学习 Demo
# ==========================================
# 安装: pip install -U langgraph
# 场景: 模拟一个"问题分类器" —— 根据问题类型
#       走不同的处理节点，最后汇总答案。
# ==========================================

from typing import Annotated, Literal
from typing_extensions import TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


# ──────────────────────────────────────────
# 第一步：定义 State（状态/共享小黑板）
# ──────────────────────────────────────────
class State(TypedDict):
    # 用户输入的问题
    question: str
    # 问题类型（将在节点中填充）
    question_type: str
    # 消息列表：使用 Annotated + add 作为 reducer
    # 这意味着每次更新是"追加"而非"覆盖"
    messages: Annotated[list[str], add]
    # 最终答案
    answer: str


# ──────────────────────────────────────────
# 第二步：定义 Nodes（节点）
# 每个节点就是一个接收 state 的普通函数
# ──────────────────────────────────────────

def classify_node(state: State) -> dict:
    """
    分类节点：判断问题属于哪种类型。
    相当于"工厂大门口的分拣员"。
    """
    question = state["question"]
    print(f"[分类节点] 收到问题: {question}")

    # 简单关键词分类（实际可换成 LLM 判断）
    if "天气" in question or "weather" in question:
        q_type = "weather"
    elif "代码" in question or "code" in question or "python" in question.lower():
        q_type = "code"
    else:
        q_type = "general"

    return {
        "question_type": q_type,
        "messages": [f"✅ 分类完成：问题类型 = {q_type}"],
    }


def weather_node(state: State) -> dict:
    """处理天气类问题的节点"""
    print(f"[天气节点] 处理天气问题")
    answer = f"今天天气晴好，气温 22°C！（回答了问题：{state['question']}）"
    return {
        "answer": answer,
        "messages": ["🌤️ 天气节点已处理"],
    }


def code_node(state: State) -> dict:
    """处理代码类问题的节点"""
    print(f"[代码节点] 处理代码问题")
    answer = f"关于代码问题：建议先查文档，再 Google，再问 AI。（问题：{state['question']}）"
    return {
        "answer": answer,
        "messages": ["💻 代码节点已处理"],
    }


def general_node(state: State) -> dict:
    """处理通用问题的节点"""
    print(f"[通用节点] 处理通用问题")
    answer = f"这是一个通用回答：{state['question']}——这个问题很有趣！"
    return {
        "answer": answer,
        "messages": ["💬 通用节点已处理"],
    }


def summary_node(state: State) -> dict:
    """汇总节点：所有路径都汇聚到这里"""
    print(f"[汇总节点] 生成最终摘要")
    summary = f"【最终答案】{state['answer']}\n【处理日志】{' → '.join(state['messages'])}"
    return {
        "messages": ["📋 汇总完成"],
        "answer": summary,
    }


# ──────────────────────────────────────────
# 第三步：定义路由函数（用于条件边）
# ──────────────────────────────────────────

def route_question(state: State) -> Literal["weather_node", "code_node", "general_node"]:
    """
    根据 question_type 决定下一步走哪个节点。
    这就是"条件边"的路由函数。
    """
    q_type = state["question_type"]
    if q_type == "weather":
        return "weather_node"
    elif q_type == "code":
        return "code_node"
    else:
        return "general_node"


# ──────────────────────────────────────────
# 第四步：构建图（把节点和边组装起来）
# ──────────────────────────────────────────

builder = StateGraph(State)

# 添加节点
builder.add_node("classify_node", classify_node)
builder.add_node("weather_node", weather_node)
builder.add_node("code_node", code_node)
builder.add_node("general_node", general_node)
builder.add_node("summary_node", summary_node)

# 添加普通边：START → 分类节点
builder.add_edge(START, "classify_node")

# 添加条件边：分类节点 → 根据类型选择路径
builder.add_conditional_edges(
    "classify_node",       # 从哪个节点出发
    route_question,        # 路由函数
    {                      # 返回值 → 节点名 的映射
        "weather_node": "weather_node",
        "code_node": "code_node",
        "general_node": "general_node",
    }
)

# 普通边：三个处理节点都汇聚到汇总节点
builder.add_edge("weather_node", "summary_node")
builder.add_edge("code_node", "summary_node")
builder.add_edge("general_node", "summary_node")

# 普通边：汇总节点 → END
builder.add_edge("summary_node", END)

# 编译图（必须执行，才能使用）
graph = builder.compile()


# ──────────────────────────────────────────
# 第五步：运行图
# ──────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "今天天气怎么样？",
        "如何用 Python 写一个快速排序？",
        "为什么天空是蓝色的？",
    ]

    for q in test_questions:
        print("\n" + "=" * 50)
        print(f"❓ 问题: {q}")
        print("=" * 50)

        result = graph.invoke({
            "question": q,
            "question_type": "",
            "messages": [],
            "answer": "",
        })

        print(f"\n{result['answer']}")