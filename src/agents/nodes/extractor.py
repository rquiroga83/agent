

from agents.state import State
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage

class ContactInfo(BaseModel):   
    name: str = Field(description="The name of the person")
    phone: str = Field(description="The phone number of the person")
    email: str = Field(description="The email address of the person")
    tone: int = Field(description="The tone of the message, either formal or informal", ge=0, le=100)
    age: int = Field(description="The age of the person")
    sentiment: str = Field(description="The sentiment of the message, either positive, negative, or neutral")

model = ChatDeepSeek(model="deepseek-chat", temperature=0.0)

SYSTEM_PROMPT = """Eres un asistente que extrae informacion de conversaciones con clientes.
Extrae el nombre del cliente, su numero de telefono, su correo electronico, la edad del cliente,
el tono del mensaje (0-100) y el sentimiento del mensaje (positivo, negativo o neutral)."""

def extractor_node(state: State) :
    new_state: State = {}   # type: ignore
    history = state["messages"]

    if state.get("customer_name") is None:
        model_whith_schema = model.with_structured_output(schema=ContactInfo)
        contact_info= model_whith_schema.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history)

        new_state["customer_name"] = contact_info.name # type: ignore
        new_state["phone"] = contact_info.phone # type: ignore
        new_state["email"] = contact_info.email # type: ignore
        new_state["tone"] = contact_info.tone # type: ignore
        new_state["age"] = contact_info.age # type: ignore
        new_state["sentiment"] = contact_info.sentiment # type: ignore


    return new_state