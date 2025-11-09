from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_client import QdrantClient

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):   
    name: str = Field(description="The name of the person")
    phone: str = Field(description="The phone number of the person")
    email: str = Field(description="The email address of the person")
    tone: int = Field(description="The tone of the message, either formal or informal", ge=0, le=100)
    age: int = Field(description="The age of the person")
    sentiment: str = Field(description="The sentiment of the message, either positive, negative, or neutral")


class State(MessagesState):
    customer_name: str
    phone: str
    email: str
    tone: int
    age: int
    sentiment: str


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


def extractor_node(state: State) :
    new_state: State = {}   # type: ignore
    history = state["messages"]

    if state.get("customer_name") is None:
        llm_extractor = ChatDeepSeek(model="deepseek-chat", temperature=0.0)
        model_whith_schema = llm_extractor.with_structured_output(schema=ContactInfo)
        contact_info= model_whith_schema.invoke(history)

        new_state["customer_name"] = contact_info.name # type: ignore
        new_state["phone"] = contact_info.phone # type: ignore
        new_state["email"] = contact_info.email # type: ignore
        new_state["tone"] = contact_info.tone # type: ignore
        new_state["age"] = contact_info.age # type: ignore
        new_state["sentiment"] = contact_info.sentiment # type: ignore


    return new_state


def conversation_node(state: State) :
    new_state: State = {}   # type: ignore
    history = state["messages"]
    last_message = history[-1].content
    
    # Recuperar documentos relevantes de Qdrant
    docs = retriever.invoke(last_message) # type: ignore
    context = "\n\n".join([doc.page_content for doc in docs])

     # Agregar contexto al estado
    history = [SystemMessage(content=f"Contexto relevante:\n{context}"), 
               SystemMessage(content=f"El nombre del cliente es {state['customer_name']}.")
            ]
 

    
    ia_msg = model.invoke(history)
    new_state["messages"] = [ia_msg]

    return new_state



builder = StateGraph(State)
builder.add_node("conversation_node", conversation_node)
builder.add_node("extractor_node", extractor_node)

builder.add_edge(START, "extractor_node")
builder.add_edge("extractor_node", "conversation_node")
builder.add_edge("conversation_node", END)

agent = builder.compile()


