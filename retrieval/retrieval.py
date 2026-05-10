from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

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
