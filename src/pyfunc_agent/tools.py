"""Simple python functions for agents to use.

We're aiming for examples! Not something actually useful!

"""

import numpy as np


def add_numbers(a: float, b: float) -> float:
    """Add two integers."""
    return a + b


def square_root(a: float) -> float:
    """Take the square root of a number."""
    return np.sqrt(a)


def exponential(a: float) -> float:
    """Calculate the exponential of a number."""
    return np.exp(a)


def ln(a: float) -> float:
    """Calculate natural log of a number."""
    return np.log(a)


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a*b
