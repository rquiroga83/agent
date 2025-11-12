from agents.nodes.booking import booking_node
from agents.nodes.rag import rag_node
from agents.nodes.route import route_node
from agents.state import State
from langgraph.graph import StateGraph, START, END, MessagesState
from agents.nodes.conversation import conversation_node
from agents.nodes.extractor import extractor_node
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage


def should_continue(state: State):
    """Decide si continuar ejecutando herramientas o terminar."""
    last_message = state["messages"][-1]
    
    # Si el último mensaje tiene tool_calls, ir al nodo de herramientas
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:  # type: ignore
        print(last_message.tool_calls)  # Debug: Print tool calls
        return "tools"
    
    # Si no, terminar
    return "end"


builder = StateGraph(State)
builder.add_node("conversation_node", conversation_node)
builder.add_node("extractor_node", extractor_node)
builder.add_node("rag_node", rag_node)
builder.add_node("booking_node", booking_node)

builder.add_edge(START, "extractor_node")
builder.add_conditional_edges("extractor_node", route_node, {
    "conversation": "conversation_node",
    "booking": "booking_node"
})
#builder.add_edge("extractor_node", "conversation_node")
builder.add_conditional_edges(
    "conversation_node",
    should_continue,
    {
        "tools": "rag_node",
        "end": END
    }
)
builder.add_edge("rag_node", "conversation_node")

agent = builder.compile()