from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable


from local_llm.chat_models import get_local_model
from schema import SearchRequired, State

def requires_search(state: State) -> str:

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a routing assistant. Your only job is to evaluate the conversation and decide if a search tool is needed to accurately answer the user's latest message."),
        MessagesPlaceholder(variable_name="conversation_history"),
    ])

    model = get_local_model(model_name="llama3.2")
    structured_model: Runnable = model.with_structured_output(SearchRequired)

    chain = prompt | structured_model

    response: SearchRequired = chain.invoke({"conversation_history": state["messages"][-3:]})

    return "search_required" if response.requires_search else "no_search_required"

