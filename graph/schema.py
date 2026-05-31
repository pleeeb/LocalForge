from langchain_core.messages import AnyMessage
from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    number_of_calls: int
