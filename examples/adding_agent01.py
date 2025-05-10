"""An agent that can call an addition function."""

import sys

from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START
#from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
#from langchain_core.messages import SystemMessage
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
    #model="gemma3:12b-it-qat",  # Error, can't use tools
    #model="mistral",  # No error, no tool use though doesn't use JSON.
    model="mix_77/gemma3-qat-tools:12b",  # Works, calls the tool
    temperature=0.0,
)

# Bind tool
llm = llm.bind_tools([add_tool])


# --- AGENT NODE ---
def agent_node(state: dict) -> dict:
    """Message passer for LLM."""
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": messages + [response]}


# --- LANGGRAPH WORKFLOW ---
# Use the ToolNode to handle tool calls dynamically
tool_node = ToolNode([add_tool])

builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("agent", RunnableLambda(agent_node))
builder.add_node("tools", tool_node)

# Add edges
builder.add_edge(START, "agent")
# If the last AIMessage has tool_calls go to "tools", otherwise END
builder.add_conditional_edges("agent", tools_condition)
# After "tools" run, send result back through agent
builder.add_edge("tools", "agent")

# # Alternative linear graph
# builder.set_entry_point("agent")
# builder.add_edge("agent", "tools")
# builder.add_edge("tools", END)

# Compile the graph
graph = builder.compile()


# --- CLI HANDLER ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input string.\n"
              "Example: Add 4 and 5.2")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])

    # Set up a system prompt to make LLM aware of tool and format it's responses
    # accordingly
    # system = SystemMessage(
    #     content=(
    #         "You are an agent with access to a single tool:\n"
    #         "- add_tool(a: float, b: float) -> returns sum of a and b\n"
    #         "Whenever you are asked to add two numbers, you MUST call add_tool.\n"
    #         "**DO NOT COMPUTE THE SUM YOURSELF**\n"
    #         "\nFormat your response **exactly** as a JSON function-call, e.g.:\n"
    #         "{'name': 'add_tool', 'arguments': {'a': float, 'b': float}}"
    #     )
    # )

    input_state = {
        "messages": [
            #system,
            HumanMessage(content=user_input)
            ]
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
