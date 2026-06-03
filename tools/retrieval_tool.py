from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from retrieval.retrieval import DocumentRetrievalManager

def create_retrieval_tool(retriever: DocumentRetrievalManager):

    @tool(response_format="content_and_artifact")
    def retrieve_documents(query: str, config: RunnableConfig) -> tuple[str, list[str]]:
        """
            Searches the user's private, local document vault. 
            Use this tool WHENEVER the user asks about their own uploaded files, personal data, or specific project context. 
            If the user asks a factual question and you are not 100% certain of the answer from your base training, 
            you MUST use this tool to check the local vault before answering. The vault contains a wide variety of user-provided formats (PDFs, text, code).
        """
        top_k = config.get("configurable", {}).get("top_k", 5)
        retrieved_nodes = retriever.get_documents(query=query, top_k=top_k)
        source_chips = []
        
        results = []
        for node in retrieved_nodes:
            source_file = node.metadata.get("file_name", "Unknown Source")
            text = f"Source: {source_file}\nText: {node.text}"
            results.append(text)

            chip_text = f"{source_file}: {node.metadata.get('file_path', 'Unknown Source')}"
            if chip_text not in source_chips:
                source_chips.append(chip_text)

        content = "\n\n---\n\n".join(results)

        return content, source_chips
    
    return retrieve_documents
