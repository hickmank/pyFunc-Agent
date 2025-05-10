"""An agent that can call an addition function."""

import sys

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from pyfunc_agent.tools import add_numbers


# --- TOOL WRAPPER ---
@tool
def add_tool(a: float, b: float) -> str:
    """The tool wrapper for the agent."""
    print(f"[add_tool] called with a={a}, b={b}.")
    return add_numbers(a, b)


# --- LLM SETUP ---
llm = ChatOllama(
    model="gemma3:12b-it-qat",
    temperature=0.1,
)


# --- AGENT NODE ---
def agent_node(state):
    """Message passer for LLM."""
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": messages + [response]}


# Use the ToolNode to handle tool calls dynamically
tool_node = ToolNode([add_tool])

# --- LANGGRAPH WORKFLOW ---
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", RunnableLambda(agent_node))
builder.add_node("tools", tool_node)

# Add edges
builder.set_entry_point("agent")
builder.add_edge("agent", "tools")
builder.add_edge("tools", END)

# Compile the graph
graph = builder.compile()


# --- CLI HANDLER ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input string.\n"
              "Example: Add 4 and 5.2")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])

    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    result = graph.invoke(input_state)

    print("RESULT:")
    print(result)
    print("=====================================")

    print("Final Output:")
    print(result["messages"][-1])

    print("============ Message Chain ==========")
    for msg in result["messages"]:
        print(msg)
