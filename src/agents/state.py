from langgraph.graph import MessagesState

class State(MessagesState):
    customer_name: str
    phone: str
    email: str
    tone: int
    age: int
    sentiment: str