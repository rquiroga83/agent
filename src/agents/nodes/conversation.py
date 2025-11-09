from agents.state import State
from agents.tools.rag import buscar_optimizacion_web_mobile
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.prompts import SystemMessagePromptTemplate
import datetime
from langchain.agents import create_agent

#model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)
model = ChatDeepSeek(model="deepseek-reasoner", temperature=1.0)
model_with_tools = model.bind_tools([buscar_optimizacion_web_mobile])

SYSTEM_PROMPT = """\
Eres un asistente que da informacion sobre programacion a los usuarios.
{% if customer_name %}
El nombre del cliente es {{ customer_name }}.
{% else %}
No se ha proporcionado el nombre del cliente. solicita amablemente su nombre.
{% endif -%}

Cuando el usuario pregunte sobre temas relacionados con optimización web móvil, rendimiento, 
mejores prácticas, o cualquier aspecto técnico de sitios web móviles, DEBES usar la herramienta 
'buscar_optimizacion_web_mobile' para obtener información específica de la base de datos vectorial.

Formula queries claros y específicos para la herramienta, enfocándote en los conceptos clave de la pregunta.

Para preguntas generales o de conversación, responde directamente sin usar herramientas.
"""


def conversation_node(state: State) :
    history = state["messages"]

    prompt_template = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT, template_format="jinja2")
    system_prompt = prompt_template.format( customer_name=state.get("customer_name") )
    messages = [system_prompt] + history
 
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


"""
tools=[buscar_optimizacion_web_mobile]

conversation_node  = create_agent(
    model, 
    tools=tools,
    system_prompt="Eres un asistente que da informacion sobre programacion a los usuarios."
)
"""

