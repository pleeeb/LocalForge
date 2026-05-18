from llama_index.core import SimpleDirectoryReader, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline

from llama_index.readers.file import PyMuPDFReader

from ingestion.title_extractor import StructuredTitleExtractor
from local_llm.llm import OllamaLLM
from vector_store.chroma import VectorStoreProvider

class DocumentPipelineManager:
    def __init__(self, provider: VectorStoreProvider):
        llm = OllamaLLM(model_name="llama3.2").get_model()
        
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                StructuredTitleExtractor(llm=llm),
                provider.embed_model
            ],
            vector_store=provider.vector_store,
        )

    def process_directory(self, directory_path: str, multi_files: bool = True):
        allowed_extensions = ['.txt', '.pdf', '.docx']

        pdf_extractor = PyMuPDFReader()

        reader = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            required_exts=allowed_extensions,
            recursive=multi_files,
            file_extractor={'.pdf': pdf_extractor})

        return self.pipeline.run(documents=reader.load_data())
        