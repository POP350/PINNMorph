"""Online, in-training evolution for physics-informed neural networks."""

from .pipeline.a_problem_definition import get_problem, list_problems

__version__ = "0.1.0"

__all__ = [
    "get_problem",
    "list_problems",
]
