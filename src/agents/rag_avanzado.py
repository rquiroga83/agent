from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient

import random

class State(MessagesState):
    customer_name: str

# Configurar cliente Qdrant
qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "class_collection"

# Configurar embeddings con Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Crear vector store
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings,
)

# Crear retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Definir herramienta de búsqueda vectorial
@tool
def buscar_optimizacion_web_mobile(query: str) -> str:
    """Busca información sobre optimización de sitios web móviles en la base de datos.
    
    Usa esta herramienta cuando el usuario pregunte sobre:
    - Optimización de rendimiento web móvil
    - Mejores prácticas para sitios web móviles
    - Técnicas de optimización mobile
    - Performance web en dispositivos móviles
    - SEO móvil y experiencia de usuario
    
    Args:
        query: Una consulta específica sobre optimización web móvil
    
    Returns:
        Información relevante encontrada en la base de datos
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante sobre optimización web móvil."
    
    context = "\n\n".join([f"📄 Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return f"Información sobre optimización web móvil:\n\n{context}"

# Modelo con herramientas
model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
model_with_tools = model.bind_tools([buscar_optimizacion_web_mobile])

def agent_node(state: State):
    """Nodo del agente que decide si usar herramientas."""
    history = state["messages"]
    
    # Sistema de instrucciones
    system_prompt = SystemMessage(content="""Eres un asistente experto en optimización de sitios web móviles.

Cuando el usuario pregunte sobre temas relacionados con optimización web móvil, rendimiento, 
mejores prácticas, o cualquier aspecto técnico de sitios web móviles, DEBES usar la herramienta 
'buscar_optimizacion_web_mobile' para obtener información específica de la base de datos.

Formula queries claros y específicos para la herramienta, enfocándote en los conceptos clave de la pregunta.

Para preguntas generales o de conversación, responde directamente sin usar herramientas.""")
    
    messages = [system_prompt] + history
    response = model_with_tools.invoke(messages)
    
    return {"messages": [response]}

def tool_node(state: State):
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

def should_continue(state: State):
    """Decide si continuar ejecutando herramientas o terminar."""
    last_message = state["messages"][-1]
    
    # Si el último mensaje tiene tool_calls, ir al nodo de herramientas
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:  # type: ignore
        return "tools"
    
    # Si no, terminar
    return "end"

# Construir el grafo
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
builder.add_edge("tools", "agent")

agent = builder.compile()


