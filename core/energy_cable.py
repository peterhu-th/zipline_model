from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from config.params import CableParams, PhysicalConstants, SolverParams


@dataclass
class CableShape:
    x: np.ndarray
    y: np.ndarray
    tension: np.ndarray
    lengths: np.ndarray
    rest_lengths: np.ndarray
    success: bool
    objective: float
    message: str = ""

    @property
    def T_max(self) -> float:
        return float(np.max(self.tension)) if self.tension.size else 0.0

    @property
    def arc_length(self) -> float:
        return float(np.sum(self.lengths))


def make_cable_params(base: CableParams, **updates: float | int | bool) -> CableParams:
    return replace(base, **updates)


def initial_shape(cable: CableParams, sag_ratio: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, cable.W, cable.N + 1)
    chord = cable.H * (1.0 - x / cable.W)
    sag = sag_ratio * cable.W * np.sin(np.pi * x / cable.W)
    y = chord - sag
    return x, y


def _pack(x: np.ndarray, y: np.ndarray, fixed_x: bool) -> np.ndarray:
    if fixed_x:
        return y[1:-1].copy()
    return np.column_stack([x[1:-1], y[1:-1]]).ravel()


def _unpack(q: np.ndarray, cable: CableParams) -> tuple[np.ndarray, np.ndarray]:
    if cable.fixed_x:
        x = np.linspace(0.0, cable.W, cable.N + 1)
        y = np.empty(cable.N + 1)
        y[0], y[-1] = cable.H, 0.0
        y[1:-1] = q
        return x, y
    nodes = q.reshape(-1, 2)
    x = np.concatenate([[0.0], nodes[:, 0], [cable.W]])
    y = np.concatenate([[cable.H], nodes[:, 1], [0.0]])
    return x, y


def _interp_piecewise(x: np.ndarray, y: np.ndarray, xp: float) -> float:
    xp = float(np.clip(xp, x[0], x[-1]))
    return float(np.interp(xp, x, y))


def total_potential_energy(
    q: np.ndarray,
    cable: CableParams,
    const: PhysicalConstants,
    rider_mass: Optional[float] = None,
    rider_x: Optional[float] = None,
) -> float:
    x, y = _unpack(q, cable)
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        return 1e30 + 1e24 * float(np.sum(np.minimum(dx, 0.0) ** 2))

    lengths = np.hypot(dx, np.diff(y))
    l0 = np.full(cable.N, cable.L / cable.N)
    stretch = np.maximum(lengths - l0, 0.0)

    cable_gravity = np.sum(cable.R * l0 * const.g * (y[:-1] + y[1:]) / 2.0)
    elastic = np.sum(cable.EA * stretch * stretch / (2.0 * l0))
    rider = 0.0
    if rider_mass is not None and rider_x is not None:
        rider = float(rider_mass) * const.g * _interp_piecewise(x, y, rider_x)
    return float(cable_gravity + elastic + rider)


def compute_tension(x: np.ndarray, y: np.ndarray, cable: CableParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.hypot(np.diff(x), np.diff(y))
    l0 = np.full(cable.N, cable.L / cable.N)
    strain = (lengths - l0) / l0
    tension = cable.EA * np.maximum(strain, 0.0)
    return tension, lengths, l0


def solve_cable_shape(
    cable: CableParams,
    const: Optional[PhysicalConstants] = None,
    solver: Optional[SolverParams] = None,
    rider_mass: Optional[float] = None,
    rider_x: Optional[float] = None,
    initial: Optional[CableShape] = None,
) -> CableShape:
    const = const or PhysicalConstants()
    solver = solver or SolverParams()

    if initial is None or initial.x.size != cable.N + 1:
        x0, y0 = initial_shape(cable)
    else:
        x0, y0 = initial.x.copy(), initial.y.copy()
        x0[0], y0[0], x0[-1], y0[-1] = 0.0, cable.H, cable.W, 0.0

    q0 = _pack(x0, y0, cable.fixed_x)
    bounds = None
    constraints = []
    if cable.fixed_x:
        y_pad = max(cable.W, abs(cable.H), cable.L)
        bounds = [(min(0.0, cable.H) - y_pad, max(0.0, cable.H) + y_pad)] * (cable.N - 1)
    else:
        y_pad = max(cable.W, abs(cable.H), cable.L)
        bounds = []
        eps = 1e-5
        for _ in range(cable.N - 1):
            bounds.append((eps, cable.W - eps))
            bounds.append((min(0.0, cable.H) - y_pad, max(0.0, cable.H) + y_pad))

        def ordered_nodes(q: np.ndarray) -> np.ndarray:
            x, _ = _unpack(q, cable)
            return np.diff(x) - eps

        constraints.append({"type": "ineq", "fun": ordered_nodes})

    result = minimize(
        total_potential_energy,
        q0,
        args=(cable, const, rider_mass, rider_x),
        method=solver.optimizer_method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": solver.max_iter, "ftol": solver.ftol, "disp": False},
    )
    x, y = _unpack(result.x if result.x is not None else q0, cable)
    tension, lengths, l0 = compute_tension(x, y, cable)
    return CableShape(
        x=x,
        y=y,
        tension=tension,
        lengths=lengths,
        rest_lengths=l0,
        success=bool(result.success),
        objective=float(result.fun) if result.fun is not None else float("nan"),
        message=str(result.message),
    )

