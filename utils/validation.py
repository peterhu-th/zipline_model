from __future__ import annotations
from config.params import CableParams


def validate_cable_params(cable: CableParams) -> None:
    if cable.W <= 0:
        raise ValueError("W must be positive.")
    if cable.L <= (cable.W * cable.W + cable.H * cable.H) ** 0.5:
        raise ValueError("L must exceed the straight distance between fixed points.")
    if cable.R <= 0 or cable.E <= 0 or cable.A_c <= 0:
        raise ValueError("R, E, and A_c must be positive.")
    if cable.N < 3:
        raise ValueError("N must be at least 3.")
