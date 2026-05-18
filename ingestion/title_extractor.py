from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field
from llama_index.core.schema import TransformComponent, TextNode
from typing import Any

class TitleSchema(BaseModel):
    title: str

class StructuredTitleExtractor(TransformComponent):
    llm: Any = Field(description="The LLM used for structured prediction")

    def __call__(self, nodes, **kwargs):
        print(f"Extracting title from {len(nodes)} nodes.")

        seen_docs = {}

        for node in nodes:
            if isinstance(node, TextNode):

                doc_id = node.metadata.get("file_name", "unknown_doc")

                if doc_id in seen_docs:
                    node.metadata["document_title"] = seen_docs[doc_id]
                    continue

                title_prompt = PromptTemplate(
                    "Extract a highly concise title for the following text. "
                    "Do not include any conversational filler. \n\n"
                    "Text:\n{context_str}"
                )

                try:
                    result = self.llm.structured_predict(TitleSchema, prompt=title_prompt, context_str=node.text[:1000])

                    clean_title = result.title
                    node.metadata["document_title"] = clean_title
                    seen_docs[doc_id] = clean_title
                except Exception as e:
                    print(f"Failed to parse JSON for {doc_id}. Fallback to file name. Error: {e}")
                    fallback_title = node.metadata.get("file_name", "Unknown Title")
                    node.metadata["document_title"] = fallback_title
                    seen_docs[doc_id] = fallback_title

        return nodes