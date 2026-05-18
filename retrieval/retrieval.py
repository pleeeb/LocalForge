from llama_index.core import VectorStoreIndex

from vector_store.chroma import VectorStoreProvider

class DocumentRetrievalManager:
    def __init__(self, provider: VectorStoreProvider, default_top_k: int = 5):
        self.index = VectorStoreIndex.from_vector_store(
            provider.vector_store,
            embed_model=provider.embed_model
        )

        self.retriever = self.index.as_retriever(similarity_top_k=default_top_k)

    def get_documents(self, query: str, top_k: int = 5):
        return self.retriever.retrieve(query)
    
    def print_retrieved_information(self, nodes):
        for i, node in enumerate(nodes):
            print(f"Node {i+1}:")
            print(f"Score: {node.score:.4f}")
            print(f"Text: {node.text}")
            print(f"Metadata: {node.metadata}")
            print("-" * 40)

query = "What are Peters accomplishments in computer science?"

manager = DocumentRetrievalManager(provider=VectorStoreProvider(collection_name="test_collection"))
retrieved_nodes = manager.get_documents(query=query, top_k=1)
manager.print_retrieved_information(retrieved_nodes)