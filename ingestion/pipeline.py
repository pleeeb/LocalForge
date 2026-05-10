from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

from .ingest import ingest_from_directory

class DocumentPipelineManager:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model_name: str = "all-MiniLM-L6-v2"):
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                TitleExtractor(),
                HuggingFaceEmbedding(model_name=model_name)
            ]
        )

    def process_directory(self, directory_path: str, multi_files: bool = True):
        documents = ingest_from_directory(directory_path, multi_files)

        processed_documents = self.pipeline.run(documents)
        
        return processed_documents