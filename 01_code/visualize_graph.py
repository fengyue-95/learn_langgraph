from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)


def mock_llm(state: MessagesState):
    result = model.invoke(state["messages"])
    return {"messages": [result]}


graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
agent = graph.compile()

# 方法1: 打印图的ASCII表示
print("图结构 (ASCII):")
print(agent.get_graph().draw_ascii())

print("\n" + "="*50 + "\n")

# 方法2: 获取图的Mermaid格式（可以在线渲染）
print("Mermaid格式 (可复制到 https://mermaid.live 查看):")
print(agent.get_graph().draw_mermaid())

print("\n" + "="*50 + "\n")

# 方法3: 保存为PNG图片（需要安装 pygraphviz 或 graphviz）
try:
    from IPython.display import Image
    png_data = agent.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("图结构已保存到 graph.png")
except Exception as e:
    print(f"保存PNG失败（需要安装相关依赖）: {e}")

print("\n" + "="*50 + "\n")

# 方法4: 打印节点和边的信息
print("节点列表:")
for node in agent.get_graph().nodes:
    print(f"  - {node}")

print("\n边列表:")
for edge in agent.get_graph().edges:
    print(f"  - {edge}")
