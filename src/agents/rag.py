from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

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
model = ChatDeepSeek(model="deepseek-chat", temperature=1.0)



def node_1(state: State) :
    new_state: State = {}   # type: ignore

    if state.get("customer_name") is None:
        new_state["customer_name"] = "Alice"
        new_state["messages"] = [
            HumanMessage(content=f"Mi mobre es {new_state['customer_name']}.")
        ]

    

    history = state["messages"]
    last_message = history[-1].content
    
    # Recuperar documentos relevantes de Qdrant
    docs = retriever.invoke(last_message) # type: ignore
    context = "\n\n".join([doc.page_content for doc in docs])

     # Agregar contexto al estado
    new_state["messages"] = [SystemMessage(content=f"Contexto relevante:\n{context}"), HumanMessage(content=last_message)]
    

    ia_msg = model.invoke(history)
    new_state["messages"] = [ia_msg]

    return new_state



builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

agent = builder.compile()


