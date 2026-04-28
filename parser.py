from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


def parse_message(message: BaseMessage) -> Dict[str, Any]:
    """解析单个消息对象为JSON格式"""
    return {
        "role": message.type,
        "content": message.content,
        "id": message.id,
        "additional_kwargs": message.additional_kwargs,
        "response_metadata": getattr(message, "response_metadata", {}),
        "tool_calls": getattr(message, "tool_calls", []),
        "usage_metadata": getattr(message, "usage_metadata", {})
    }


def parse_result(result: Dict[str, List[BaseMessage]]) -> Dict[str, Any]:
    """解析完整的LangGraph结果为JSON格式"""
    messages = result.get("messages", [])

    parsed_messages = []
    for msg in messages:
        parsed_messages.append(parse_message(msg))

    # 提取最后一条AI消息作为主要响应
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    return {
        "messages": parsed_messages,
        "last_response": {
            "content": last_ai_message.content if last_ai_message else None,
            "usage": getattr(last_ai_message, "usage_metadata", {}) if last_ai_message else {}
        },
        "total_messages": len(messages)
    }


def get_last_ai_content(result: Dict[str, List[BaseMessage]]) -> str:
    """快速获取最后一条AI回复的内容"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""
