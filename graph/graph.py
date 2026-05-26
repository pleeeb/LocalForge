from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from langgraph.graph import StateGraph

def create_graph(state_schema, context_schema) -> StateGraph:
    graph = StateGraph(state_schema=state_schema, context_schema=context_schema)

    return graph

def compile_graph(graph: StateGraph, checkpointer: Checkpointer, store: BaseStore):
    return graph.compile(checkpointer=checkpointer, store=store)