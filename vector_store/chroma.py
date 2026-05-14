import chromadb
from datetime import datetime, timezone

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class VectorStoreProvider:
    def __init__(self, collection_name: str, model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient("./db_storage/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"created_at": str(datetime.now(timezone.utc))},
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)