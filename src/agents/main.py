from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [get_current_time]

model = ChatDeepSeek(model="deepseek-chat")
agent = create_agent(
    model, 
    tools=tools,
    system_prompt="You are a helpful assistant."
)

