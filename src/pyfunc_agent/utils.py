"""Utilities for agentic systems."""

from pathlib import Path
import json
import yaml

# If `simple_agents.py` lives in pyfunc_agent/, then:
PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt_json(name: str) -> str:
    """Load a JSON prompt.

    Given a filename (like "fizban.json"), open it,
    read the "description" field, and return the string.
    """
    print(f"load_prompt: {PROMPT_DIR}")
    data = json.loads((PROMPT_DIR / name).read_text(encoding="utf-8"))

    return data["description"]


def load_prompt_yaml(name: str) -> str:
    """Load YAML prompt.

    Given a filename like "calc_bot.yaml", open it, read the 'description' field,
    and return that multiline string.
    """
    raw = (PROMPT_DIR / name).read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    return payload["description"]
