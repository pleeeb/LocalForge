from ingestion.pipeline import DocumentPipelineManager
from vector_store.chroma import VectorStoreProvider

import chromadb
import json
from pathlib import Path

def create_vector_store_collection():
    provider = VectorStoreProvider("test_collection")
    pipeline_manager = DocumentPipelineManager(provider=provider)

    directory_path = "./test_files"

    pipeline_manager.process_directory(directory_path, multi_files=False)


def export_collection_to_json(collection_name: str, output_filepath: str):
    print(f"Connecting to database to export '{collection_name}'...")
    
    # 1. Connect to the client
    client = chromadb.PersistentClient("./db_storage/chroma_db")
    collection = client.get_collection(name=collection_name)

    # 2. Fetch the data
    # We explicitly tell Chroma NOT to return the embeddings array to save space.
    # We only want the human/AI readable text and metadata.
    data = collection.get(include=["metadatas", "documents"])
    
    total_records = len(data['ids'])
    print(f"Found {total_records} records. Formatting data...")

    # 3. Zip the parallel arrays into a clean list of dictionaries
    formatted_records = []
    for i in range(total_records):
        record = {
            "id": data['ids'][i],
            "metadata": data['metadatas'][i] if data['metadatas'] else {},
            "document": data['documents'][i] if data['documents'] else ""
        }
        formatted_records.append(record)

    # 4. Write to a JSON file
    output_path = Path(output_filepath)
    
    # indent=2 makes it highly readable for both humans and AI
    # ensure_ascii=False ensures special characters/quotes don't get garbled
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_records, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Successfully exported {total_records} records to {output_path.absolute()}")

if __name__ == "__main__":
    # Step 1: Ingest documents and populate the vector store
    create_vector_store_collection()
    # Replace with your actual collection name
    export_collection_to_json(
        collection_name="test_collection", 
        output_filepath="chroma_export.json"
    )