from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from langgraph.graph import END, START, StateGraph
from graph.nodes import agent_node, tools

def create_graph(state_schema, context_schema) -> StateGraph:
    graph = StateGraph(state_schema=state_schema, context_schema=context_schema)

    tool_node = ToolNode(tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )

    graph.add_edge("tools", "agent")

    return graph

def compile_graph(graph: StateGraph, checkpointer: Checkpointer, store: BaseStore):
    return graph.compile(checkpointer=checkpointer, store=store)