from agents.state import State
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)

SYSTEM_PROMPT = """Eres un asistente que da informacion sobre programacion a los usuarios."""

def conversation_node(state: State) :
    new_state: State = {}   # type: ignore
    history = state["messages"]
    
    history = [SystemMessage(content=SYSTEM_PROMPT)] + history
    history = [SystemMessage(content=f"El nombre del cliente es {state['customer_name']}.")] + history
 
    ia_msg = model.invoke(history)
    new_state["messages"] = [ia_msg]

    return new_state