
from agents.tools.rag import buscar_optimizacion_web_mobile
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from agents.state import State


def rag_node(state: State):
    """Nodo que ejecuta las herramientas llamadas."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Ejecutar todas las tool calls
    tool_messages = []
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:  # type: ignore
            if tool_call["name"] == "buscar_optimizacion_web_mobile":
                result = buscar_optimizacion_web_mobile.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
    
    return {"messages": tool_messages}