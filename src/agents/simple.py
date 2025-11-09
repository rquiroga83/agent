from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
import random

class State(MessagesState):
    customer_name: str

model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)

def node_1(state: State) :
    new_state: State = {}   # type: ignore

    if state.get("customer_name") is None:
        new_state["customer_name"] = "Alice"    
        new_state["messages"] = [
            HumanMessage(content=f"Mi mobre es {new_state['customer_name']}.")
        ]
    
    

    history = state["messages"]
    ia_msg = model.invoke(history)
    new_state["messages"] = [ia_msg]

    return new_state



builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

agent = builder.compile()


