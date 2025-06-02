"""Streamlit interface to modular agent."""

import streamlit as st

from pyfunc_agent.simple_agents import MultiToolMathAgent

# ------------------------------------------------------------------------------
# 1) INITIAL SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Modularized Agent Chat",
    layout="wide",
)
st.title("Modularized Agent")

# 1.1) Instantiate the agent once (cached in session_state)
if "agent" not in st.session_state:
    st.session_state.agent = MultiToolMathAgent(
        prompt_name="calc_bot.yaml",
        model_name="mix_77/gemma3-qat-tools:12b",
    )

# 1.2) Keep a simple `(prompt, reply)` history list
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []


# ------------------------------------------------------------------------------
# 2) CALLBACK FOR SUBMISSION
# ------------------------------------------------------------------------------
def send_callback() -> None:
    """Enter call.

    Called when the user hits Enter. Sends the new prompt to the agent,
    appends (prompt, reply) to history, and clears the input box.
    """
    prompt = st.session_state.user_input.strip()
    if not prompt:
        return

    # 2.1) Call our agent’s .chat() method
    reply = st.session_state.agent.chat(prompt)

    # 2.2) Store the pair
    st.session_state.history.append((prompt, reply))

    # 2.3) Clear the input box
    st.session_state.user_input = ""


# ------------------------------------------------------------------------------
# 3) RENDER FULL HISTORY + INPUT BOX
# ------------------------------------------------------------------------------
# 3.1) Show each (You: …, Agent: …) exchange
for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Agent:** {bot_msg}")
    st.write("---")

# 3.2) The TextInput stays beneath the latest conversation
st.text_input(
    "Your question for Agent:",
    key="user_input",
    placeholder="e.g. What to do next?",
    on_change=send_callback,
)
