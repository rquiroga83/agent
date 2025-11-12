from typing import Literal
from agents.state import State
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain_core.prompts import SystemMessagePromptTemplate

class RouteIntent(BaseModel):
    """Modelo de datos para representar la intención de ruta."""
    step: Literal['conversation', 'booking'] = Field (
        'conversation',
        description="El paso al que se debe dirigir el agente: 'conversation' o 'booking'."
    )

SYSTEM_PROMPT = """booking: si el usurios quiere agendar un cita 
conversation: si el usuario quiere conversar o hacer preguntas generales
"""

llm = ChatDeepSeek(model="deepseek-reasoner", temperature=1.0)
model_with_schema = llm.with_structured_output(schema=RouteIntent)

def route_node(state: State) -> Literal['conversation', 'booking']:
    """Nodo que determina la ruta del agente basado en la intención."""
    history = state["messages"]
    
    prommt_template = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT, template_format="jinja2")
    route_intent = model_with_schema.invoke([prommt_template.format()] + history)

    
    return route_intent.step # type: ignore