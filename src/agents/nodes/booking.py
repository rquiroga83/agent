from agents.state import State
from agents.tools.rag import buscar_optimizacion_web_mobile
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
import datetime
from langchain.agents import create_agent

#model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)
model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)
model_with_tools = model.bind_tools([buscar_optimizacion_web_mobile])

SYSTEM_PROMPT = """\
Eres un asistente para agendar citas si falta informacion debes solicitar fecha y hora de la cita.
"""


def booking_node(state: State) :
    history = state["messages"]

    prompt_template = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT, template_format="jinja2")
    system_prompt = prompt_template.format()
    messages = [system_prompt] + history
 
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


