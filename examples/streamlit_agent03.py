"""A streamlit-UI agent that can call multiple python functions.

Run with `>> streamlit run streamlit_agent03.py`

"""

import streamlit as st
import sys

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

# Import exactly the same functions & AgentState type as in multitool_agent02.py
from pyfunc_agent.tools import add_numbers, multiply_numbers
from pyfunc_agent.tools import square_root, exponential, ln
from pyfunc_agent.agent_attributes import AgentState


# --------------------------------------------------------------------------------
# 1) Re‐declare every @tool exactly as in multitool_agent02.py
# --------------------------------------------------------------------------------

@tool
def add_tool(a: float, b: float) -> float:
    """Returns a + b."""
    print(f"[add_tool] called with a={a}, b={b}.")
    return add_numbers(a, b)

@tool
def multiply_tool(a: float, b: float) -> float:
    """Returns a * b."""
    print(f"[multiply_tool] called with a={a}, b={b}.")
    return multiply_numbers(a, b)

@tool
def sqrt_tool(a: float) -> float:
    """Returns sqrt(a)."""
    print(f"[sqrt_tool] called with a={a}.")
    return square_root(a)

@tool
def exp_tool(a: float) -> float:
    """Returns exp(a)."""
    print(f"[exp_tool] called with a={a}.")
    return exponential(a)

@tool
def ln_tool(a: float) -> float:
    """Returns ln(a)."""
    print(f"[ln_tool] called with a={a}.")
    return ln(a)


# --------------------------------------------------------------------------------
# 2) Instantiate the same ChatOllama + bind all five tools
# --------------------------------------------------------------------------------

llm = ChatOllama(
    model="mix_77/gemma3-qat-tools:12b",
    temperature=0.0,
)
llm = llm.bind_tools([
    add_tool,
    multiply_tool,
    sqrt_tool,
    exp_tool,
    ln_tool
])

# --------------------------------------------------------------------------------
# 3) Build the identical LangGraph graph from multitool_agent02.py
# --------------------------------------------------------------------------------

def agent_node(state: AgentState) -> AgentState:
    """Agent definition function."""
    messages = state["messages"]
    response = llm.invoke(messages)

    return {"messages": messages + [response]}


tool_node = ToolNode([
    add_tool,
    multiply_tool,
    sqrt_tool,
    exp_tool,
    ln_tool
])

builder = StateGraph(MessagesState)
builder.add_node("agent", RunnableLambda(agent_node))
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


# --------------------------------------------------------------------------------
# 4) Streamlit UI: Show only the most recent exchange, with input bar at bottom
# --------------------------------------------------------------------------------

st.set_page_config(page_title="Fizban Math Agent (Latest Only)", layout="wide")
st.title("🔮 Fizban’s Multitool Math Agent (Latest Only)")

# Initialize conversation history in session_state
if "messages" not in st.session_state:
    system_prompt = SystemMessage(
        content=(
            "You are an eccentric mathematician agent named Fizban. "
            "You have access to a few simple math tools:\n"
            "- add_tool(a: float, b: float) -> returns sum of a and b\n"
            "- multiply_tool(a: float, b: float) -> returns multiplication of a and b\n"
            "- exp_tool(a: float) -> returns natural exponentiation of a\n"
            "- sqrt_tool(a: float) -> returns square root of a\n"
            "- ln_tool(a: float) -> returns natural log of a\n\n"
            "Whenever you are asked a question use your tools to answer the question "
            "if the tools are at all relevant. You love using your tools!\n\n"
            "Form your response with a lot of whimsy and explain how you "
            "used your tools to get your response."
        )
    )
    st.session_state.messages = [system_prompt]

# We'll also keep a slot in session_state for the text_input's current value
# (this is bound via key="user_input"):
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Define a callback that runs **before** Streamlit re-renders the page,
# whenever the text_input "user_input" is changed (i.e. when the user hits Enter).
def submit_callback():
    """Run a callback."""
    prompt = st.session_state["user_input"].strip()
    if prompt:
        # 1) Append the new human question
        st.session_state.messages.append(HumanMessage(content=prompt))

        # 2) Invoke the LangGraph agent (tool call → tool output → final answer)
        input_state = {"messages": st.session_state.messages}
        result_state: AgentState = graph.invoke(input_state)

        # 3) Overwrite session_state.messages with the updated list
        st.session_state.messages = result_state["messages"]

    # 4) Clear the text_input for the next question
    st.session_state.user_input = ""


# --------------------------------------------------------------------------------
# 5) Helper to “slice” out only the latest exchange from our full history
# --------------------------------------------------------------------------------

def get_latest_slice():
    """Find the index of the **last** HumanMessage.

    Then return the sub-list from that index onward (so we only show
    “You: …”, plus all AIMessage tool calls/tool outputs/final reply that followed).
    If no HumanMessage has occurred yet, return the entire list
    (so that the first thing the user sees is the SystemMessage).
    """
    msgs = st.session_state.messages
    last_h_idx = None
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage):
            last_h_idx = i
    if last_h_idx is None:
        # No user question yet; show everything (just system)
        return msgs[:]
    else:
        # Return from the most recent user question to the end
        return msgs[last_h_idx:]


def render_latest():
    """Render exactly the slice returned by get_latest_slice().

    In order.
    - **You:** <content> for HumanMessage
    - If AIMessage has a "function_call", show **Fizban (tool call):**
    - Otherwise show **Fizban:** <content>
    - If the slice starts with a SystemMessage (only on very first run), show **System:**
    """
    slice_msgs = get_latest_slice()
    for msg in slice_msgs:
        if isinstance(msg, SystemMessage):
            st.markdown(f"**System:** {msg.content}")
        elif isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            fc = msg.additional_kwargs.get("function_call")
            if fc:
                name = fc["name"]
                args = fc["arguments"]
                tool_call_msg = ', '.join(f'{k}={v}' for k, v in args.items())
                st.markdown(
                    f"**Fizban (tool call):**  `{name}({tool_call_msg})`"
                )
            else:
                # This might be the “tool output” AIMessage or the final natural‐language
                # AIMessage; either way, we label it generically as “Fizban:”
                st.markdown(f"**Fizban:** {msg.content}")
        else:
            st.markdown(f"**{type(msg).__name__}:** {msg.content}")

# --------------------------------------------------------------------------------
# 6) Build the page: First, show the “Latest Exchange”; then, show the single text_input.
#    The text_input is bound to st.session_state["user_input"],
#    and has on_change=submit_callback.
# --------------------------------------------------------------------------------

st.subheader("Latest Exchange:")
render_latest()
st.markdown("---")

# This text_input stays directly below the latest answer. When the user hits Enter,
# submit_callback() runs (appending a HumanMessage, invoking the agent, clearing the
# input), and then Streamlit re-runs the entire script—so render_latest() will show the
# newly updated slice.
st.text_input(
    "Your question for Fizban:",
    key="user_input",
    placeholder="e.g. What is the square root of 256?",
    on_change=submit_callback
)
