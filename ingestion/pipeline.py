import os

from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.readers.file import PyMuPDFReader

from ingestion.title_extractor import StructuredTitleExtractor
from local_llm.index_models import OllamaLLM
from vector_store.chroma import VectorStoreProvider

class DocumentPipelineManager:
    def __init__(self, provider: VectorStoreProvider):
        llm = OllamaLLM(model_name="llama3.2").get_model()
        self.persist_dir = "./db_storage/pipeline_storage"

        if os.path.exists(self.persist_dir):
            self.docstore = SimpleDocumentStore.from_persist_dir(self.persist_dir)
        else:
            self.docstore = SimpleDocumentStore()
        
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                StructuredTitleExtractor(llm=llm),
                provider.embed_model
            ],
            vector_store=provider.vector_store,
            docstore = self.docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS
        )

    def process_directory(self, directory_path: str, multi_files: bool = True):
        allowed_extensions = ['.txt', '.pdf', '.docx']

        pdf_extractor = PyMuPDFReader()

        reader = SimpleDirectoryReader(
            directory_path,
            required_exts=allowed_extensions,
            recursive=multi_files,
            file_extractor={'.pdf': pdf_extractor})
        
        nodes = self.pipeline.run(documents=reader.load_data())
        self.pipeline.persist(persist_dir=self.persist_dir)

        return nodes
        