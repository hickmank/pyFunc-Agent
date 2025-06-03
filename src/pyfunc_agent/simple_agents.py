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
from pyfunc_agent.utils import load_prompt_yaml

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


@tool
def finish_tool(answer: str) -> str:
    """A no-op “Finish” tool to unify the interface. It just echoes back the answer."""
    return answer

# ------------------------------------------------------------------------------
#  AGENT CLASSES
# ------------------------------------------------------------------------------
class MultiToolMathAgent:
    """Encapsulated multi-tool math agent (Fizban)."""

    def __init__(
            self,
            prompt_name: str = "fizban.yaml",
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
        system_text = load_prompt_yaml(prompt_name)
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


class ReActMathAgent:
    """Encapsulated ReAct math agent.

    The only changes that make this a ReAct agent is the system prompt structure which
    defines the ReAct process and the chat method that returns the full reasoning trace.
    """

    def __init__(
            self,
            prompt_name: str = "react_bot.yaml",
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
                finish_tool,
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
                finish_tool,
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
        system_text = load_prompt_yaml(prompt_name)
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

    def chat(
            self,
            user_input: str,
        ) -> list[str]:
            """Full ReAct trace return.

            Send a new HumanMessage, run the LangGraph workflow, and return the full
            REACT trace.

            Instead of returning only the final AIMessage.content, this method returns a
            list of strings, each string corresponding to one step in the
            *Thought / Action / Observation / … / Final Answer* chain for that single
            query.
            """
            # 1) Record how many messages we have so far
            before_len = len(self.messages)

            # 2) Append the new question
            self.messages.append(HumanMessage(content=user_input))

            # 3) Run the LangGraph workflow
            input_state: AgentState = {"messages": self.messages}
            result_state = self.graph.invoke(input_state)

            # 4) Update the stored messages
            self.messages = result_state["messages"]

            # 5) Extract only the “new” messages (from before_len onward)
            new_msgs = self.messages[before_len:]

            # 6) Format each message as a human-readable string, preserving tool calls
            trace: list[str] = []
            for msg in new_msgs:
                if isinstance(msg, SystemMessage):
                    # (We generally won’t see a new SystemMessage here—only at init time)
                    trace.append(f"System: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    trace.append(f"You: {msg.content}")
                elif isinstance(msg, AIMessage):
                    func_call = msg.additional_kwargs.get("function_call")
                    if func_call:
                        # This AIMessage is a tool-call request
                        name = func_call["name"]
                        args = func_call["arguments"]
                        # e.g. "Action: sqrt_tool(a=256)"
                        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
                        trace.append(f"Action: {name}({arg_str})")
                    else:
                        content = msg.content.strip()
                        # Distinguish “Thought: …” vs “Observation: …” vs final answer
                        if content.startswith("Thought:"):
                            trace.append(content)  # already "Thought: <…>"
                        elif content.startswith("Observation:"):
                            trace.append(content)  # already "Observation: <…>"
                        else:
                            # Anything else is the final answer string
                            trace.append(f"Answer: {content}")
                else:
                    # Just in case some other type appears
                    text = getattr(msg, "content", "")
                    trace.append(f"{type(msg).__name__}: {text}")

            return trace
