from qdrant_client import QdrantClient
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.tools import tool

#Configurar cliente Qdrant
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
@tool(
        "buscar_optimizacion_web_mobile",
        description="""Busca información sobre optimización de sitios web móviles en la base de datos vectorial.
        Usa esta herramienta cuando el usuario pregunte sobre optimización de rendimiento web móvil,
        mejores prácticas para sitios web móviles, técnicas de optimización mobile, 
        performance web en dispositivos móviles, SEO móvil y experiencia de usuario."""
)
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
        Información relevante encontrada en la base de datos vectorial
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró información relevante sobre optimización web móvil."
    
    context = "\n\n".join([f"📄 Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    return f"Información sobre optimización web móvil:\n\n{context}"