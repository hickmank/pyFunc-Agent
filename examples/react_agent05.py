"""Streamlit UI - ReAct agent."""

import streamlit as st

from pyfunc_agent.simple_agents import ReActMathAgent


st.set_page_config(
    page_title="CalcBot ReAct Demo",
    layout="wide"
    )
st.title("CalcBot (ReAct)")

if "react_agent" not in st.session_state:
    st.session_state.react_agent = ReActMathAgent(
        prompt_name="react_bot.yaml",
        model_name="mix_77/gemma3-qat-tools:12b",
    )

if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []

def send_callback() -> None:
    """Enter-button call."""
    prompt = st.session_state.user_input.strip()
    if not prompt:
      return

    # Get full trace for this response
    trace = st.session_state.react_agent.chat(prompt)

    # Store trace as single string
    joined_trace = "\n\n".join(trace)
    st.session_state.history.append((prompt, joined_trace))
    st.session_state.user_input = ""

# Render full history
for user_msg, bot_chain_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"```\n{bot_chain_msg}\n```")
    st.write("---")

st.text_input(
    "Ask CalcBot (ReAct)…",
    key="user_input",
    placeholder="e.g. What is sqrt(625) plus ln(5)?",
    on_change=send_callback,
)
