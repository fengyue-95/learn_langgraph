from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

# 1. 定义 State 结构
class MyState(TypedDict):
    messages: Annotated[list, operator.add]   # 每次追加，不覆盖
    result: str

# 2. 定义节点
def step_a(state: MyState):
    print("执行 step_a")
    return {"messages": [{"role": "user", "content": "你好"}]}

def step_b(state: MyState):
    print("执行 step_b，调用 LLM...")
    return {"messages": [{"role": "assistant", "content": "我是 Claude！"}]}

def step_c(state: MyState):
    print("执行 step_c，格式化输出")
    return {"result": "完成"}

# 3. 构建图
builder = StateGraph(MyState)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_node("step_c", step_c)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", "step_c")
builder.add_edge("step_c", END)

# 4. 编译时传入 checkpointer（这是关键！）
checkpointer = MemorySaver()   # 内存版；生产用 SqliteSaver 或 PostgresSaver
graph = builder.compile(checkpointer=checkpointer)

# 5. 运行 —— 必须提供 thread_id，checkpointer 用它区分不同会话
config = {"configurable": {"thread_id": "thread-001"}}
result = graph.invoke({"messages": [], "result": ""}, config=config)
print("----------")

# 6. 查看所有保存的 checkpoint
for cp in graph.get_state_history(config):
    print(cp.config, cp.values)
print("----------")

# 7. 时间旅行 —— 从某个历史 checkpoint 重新运行
history = list(graph.get_state_history(config))
old_config = history[-4].config   # 倒数第二个 checkpoint
graph.invoke(None, old_config)     # 从那个状态继续执行