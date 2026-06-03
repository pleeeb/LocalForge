from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig


from local_llm.chat_models import get_local_model
from retrieval.retrieval import DocumentRetrievalManager
from graph.schema import State
from tools.retrieval_tool import create_retrieval_tool
from vector_store.chroma import VectorStoreProvider

provider = VectorStoreProvider(collection_name="test_collection")
retriever = DocumentRetrievalManager(provider=provider)
retrieval_tool = create_retrieval_tool(retriever=retriever)

tools = [retrieval_tool]

def agent_node(state: State, config: RunnableConfig) -> dict:
    selected_model = config.get("configurable", {}).get("model", "llama3.2")
    selected_temperature = config.get("configurable", {}).get("temperature", 0.2)
    model = get_local_model(model_name=selected_model, temperature=selected_temperature)
    model_with_tools = model.bind_tools(tools)

    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=5000,
        strategy="last",
        token_counter="approximate",
        allow_partial=False,
        start_on="human"
    )

    system_prompt = SystemMessage(content=(
        "You are a helpful assistant. You have access to tools. "
        "When calling a tool, you MUST pass actual string and integer values for the arguments. "
        "NEVER pass the schema definition (like {'type': 'string'})."
    ))

    messages_to_pass = [system_prompt] + trimmed_messages
    
    response = model_with_tools.invoke(messages_to_pass)

    return {"messages": [response]}
