# Step 1: Define tools and model
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langfuse.langchain import CallbackHandler

load_dotenv()

# Initialize Langfuse callback handler
langfuse_handler = CallbackHandler()

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)


def llm_call(state: dict, chat: BaseChatModel):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            chat.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"],
                config={"callbacks": [langfuse_handler]}
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
