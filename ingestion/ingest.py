from llama_index.core import Document, SimpleDirectoryReader

allowed_extensions = ['.txt', '.pdf', '.docx']

def ingest_from_directory(directory_path: str, multi_files: bool) -> list[Document]:
    if multi_files:
        reader = SimpleDirectoryReader(directory_path, required_exts=allowed_extensions, recursive=True)
    else:
        reader = SimpleDirectoryReader(directory_path, required_exts=allowed_extensions)
    return reader.load_data()