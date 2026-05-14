from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

from vector_store.chroma import VectorStoreProvider

class DocumentRetrievalManager:
    def __init__(self, provider: VectorStoreProvider):
        self.index = VectorStoreIndex.from_vector_store(
            provider.vector_store,
            embed_model=provider.embed_model
        )

    def get_retriever(self, top_k: int = 5):
        return self.index.as_retriever(similarity_top_k=top_k)

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
]

llm = Ollama(
    model="qwen3:4b",
    request_timeout=60,
    context_window=8000
)

while True:
    user_input = input("Ask a question: \n")

    message = ChatMessage(role="user", content=user_input)
    messages.append(message)

    chat_response = llm.chat(messages)

    messages.append(chat_response.message)

    print(chat_response.message.role)
    print(chat_response.message.content)
