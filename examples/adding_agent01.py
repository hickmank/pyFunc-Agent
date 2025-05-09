"""An agent that can call an addition function."""

import sys

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
#from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOllama

from pyfunc_agent.tools import add_numbers


# --- TOOL WRAPPER ---
def tool_add_numbers(inputs) -> dict:
    """The tool wrapper for the agent."""
    a = float(inputs.get("a", 0))
    b = float(inputs.get("b", 0))

    return {"result": add_numbers(a, b)}


tool_executor = ToolNode([tool_add_numbers])


# --- LLM SETUP ---
llm = ChatOllama(
    model="gemma3:12b-it-qat",
    temperature=0.1,
)

# --- AGENT NODE ---
def agent_node(state) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)

    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        invocations = [
            ToolInvocation(tool=call.name, input=call.args) for call in tool_calls
            ]
        return {
            "messages": messages + [response], 
            "invocations": invocations,
            }

    return {"messages": messages + [response]}


# --- TOOL NODE ---
def tool_node(state) -> dict:
    invocations = state.get("invocations", [])
    tool_outputs = tool_executor.invoke(invocations)

    return {"messages": state["messages"] + [tool_outputs]}


# --- LANGGRAPH WORKFLOW ---
graph = StateGraph()
graph.add_node("agent", RunnableLambda(agent_node))
graph.add_node("tools", RunnableLambda(tool_node))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", END)

runnable = graph.compile()


# --- CLI HANDLER ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an input string.\n"
              "Example: Add 4 and 5.2")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    input_state = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    result = runnable.invoke(input_state)
    print("Final Output:")
    print(result["messages"][-1])
