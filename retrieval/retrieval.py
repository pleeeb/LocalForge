from llama_index.core import VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from vector_store.chroma import VectorStoreProvider

class DocumentRetrievalManager:
    def __init__(self, provider: VectorStoreProvider, docstore_path: str = "./db_storage/pipeline_storage"):
        self.index = VectorStoreIndex.from_vector_store(
            provider.vector_store,
            embed_model=provider.embed_model
        )

        self.docstore = SimpleDocumentStore.from_persist_dir(docstore_path)

    def get_documents(self, query: str, top_k: int = 5):
        vector_retriever = self.index.as_retriever(similarity_top_k=top_k)

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=self.docstore,
            similarity_top_k=top_k,
        )

        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            llm=MockLLM(),
            similarity_top_k=top_k,
            num_queries=1,
            mode=FUSION_MODES.RECIPROCAL_RANK,
        )
        return hybrid_retriever.retrieve(query)

    def print_retrieved_information(self, nodes):
        for i, node in enumerate(nodes):
            print(f"Node {i+1}:")
            print(f"Score: {node.score:.4f}")
            print(f"Text: {node.text}")
            print(f"Metadata: {node.metadata}")
            print("-" * 40)
