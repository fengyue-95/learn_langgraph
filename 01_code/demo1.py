import json

from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START, END
from parser import parse_result

load_dotenv()

from langchain.chat_models import init_chat_model
model=init_chat_model(
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
graph = graph.compile()

result=graph.invoke({"messages": [{"role": "user", "content": "你好"}]})

parsed_result = parse_result(result)
print("完整解析结果:")
print(json.dumps(parsed_result, ensure_ascii=False, indent=2))