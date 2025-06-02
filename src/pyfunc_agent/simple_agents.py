"""Simple agents.

Classes defining simple agents. LLM instances with tools that can be instantiated in
agentic systems.

"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableLambda

from pyfunc_agent.tools import (
    add_numbers,
    multiply_numbers,
    square_root,
    exponential,
    ln,
)

from pyfunc_agent.agent_attributes import AgentState
from pyfunc_agent.utils import load_prompt

# ------------------------------------------------------------------------------
#  TOOL WRAPPERS (must be at module level so @tool can register them)
# ------------------------------------------------------------------------------
@tool
def add_tool(a: float, b: float) -> float:
    """Return a + b."""
    print(f"[add_tool] called with a={a}, b={b}.")
    return add_numbers(a, b)


@tool
def multiply_tool(a: float, b: float) -> float:
    """Return a * b."""
    print(f"[multiply_tool] called with a={a}, b={b}.")
    return multiply_numbers(a, b)


@tool
def sqrt_tool(a: float) -> float:
    """Return sqrt(a)."""
    print(f"[sqrt_tool] called with a={a}.")
    return square_root(a)


@tool
def exp_tool(a: float) -> float:
    """Return exp(a)."""
    print(f"[exp_tool] called with a={a}.")
    return exponential(a)


@tool
def ln_tool(a: float) -> float:
    """Return ln(a)."""
    print(f"[ln_tool] called with a={a}.")
    return ln(a)


# ------------------------------------------------------------------------------
#  AGENT CLASSES
# ------------------------------------------------------------------------------
class MultiToolMathAgent:
    """Encapsulated multi-tool math agent (Fizban)."""

    def __init__(
            self,
            prompt_name: str = "fizban.json",
            model_name: str = "mix_77/gemma3-qat-tools:12b"
        ) -> None:
        """Initialize tools, LLM, LangGraph workflow, and message history."""
        # 2.1) Build ChatOllama and bind all tools
        self.llm = ChatOllama(model=model_name, temperature=0.0)
        self.llm = self.llm.bind_tools(
            [
                add_tool,
                multiply_tool,
                sqrt_tool,
                exp_tool,
                ln_tool,
            ]
        )

        # 2.2) Build the same LangGraph graph as before, but using methods of this class
        self.tool_node = ToolNode(
            [
                add_tool,
                multiply_tool,
                sqrt_tool,
                exp_tool,
                ln_tool,
            ]
        )

        builder = StateGraph(MessagesState)
        # Node “agent” calls self.agent_node
        builder.add_node("agent", RunnableLambda(self.agent_node))
        builder.add_node("tools", self.tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        self.graph = builder.compile()

        # 2.3) Initialize the message history with a SystemMessage
        system_text = load_prompt(prompt_name)
        system_prompt = SystemMessage(content=system_text)

        self.messages: list[HumanMessage | SystemMessage | AIMessage] = [
            system_prompt
        ]

    def agent_node(self, state: AgentState) -> AgentState:
        """Node method.

        The “node” function for LangGraph: given a state containing messages,
        invoke the LLM and return updated messages.
        """
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": messages + [response]}

    def chat(self, user_input: str) -> str:
        """Pass HumanMessage.

        Send a new HumanMessage(user_input) to the agent, run the graph,
        and return the agent's reply text.
        """
        # 1) Append new human question
        self.messages.append(HumanMessage(content=user_input))

        # 2) Invoke the LangGraph workflow
        input_state: AgentState = {"messages": self.messages}
        result_state = self.graph.invoke(input_state)

        # 3) Update the stored messages
        self.messages = result_state["messages"]

        # 4) Extract just the final AIMessage and return its text
        last_msg = self.messages[-1]
        if isinstance(last_msg, AIMessage):
            return last_msg.content

        return ""  # In case something odd happened
