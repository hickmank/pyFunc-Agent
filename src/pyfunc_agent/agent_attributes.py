"""Classes and functions for agent attributes."""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Type for agent message."""
    messages: list[BaseMessage]
