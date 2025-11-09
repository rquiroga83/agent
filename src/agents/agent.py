from agents.state import State
from langgraph.graph import StateGraph, START, END, MessagesState
from agents.nodes.conversation import conversation_node
from agents.nodes.extractor import extractor_node


builder = StateGraph(State)
builder.add_node("conversation_node", conversation_node)
builder.add_node("extractor_node", extractor_node)

builder.add_edge(START, "extractor_node")
builder.add_edge("extractor_node", "conversation_node")
builder.add_edge("conversation_node", END)

agent = builder.compile()