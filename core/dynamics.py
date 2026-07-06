from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from config.params import CableParams, PhysicalConstants, ResistanceParams, SolverParams
from core.energy_cable import CableShape, solve_cable_shape
from core.geometry import curvature, interp_y, max_sag, slope


@dataclass
class SimulationResult:
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    accel: np.ndarray
    reached_terminal: bool
    stalled: bool
    v_max: float
    x_vmax: float
    v_terminal: float
    travel_time: float
    max_accel: float
    max_decel: float
    T_max: float
    sag_max: float
    feasible: bool
    message: str = ""


class ShapeCache:
    def __init__(self, cable: CableParams, const: PhysicalConstants, solver: SolverParams, mass: float):
        self.cable = cable
        self.const = const
        self.solver = solver
        self.mass = mass
        self.cache: dict[float, CableShape] = {}
        self.last_shape: CableShape | None = None

    def get(self, x: float) -> CableShape:
        key = round(float(np.clip(x, 0.0, self.cable.W)) / self.solver.cache_dx) * self.solver.cache_dx
        if key not in self.cache:
            self.last_shape = solve_cable_shape(
                self.cable,
                self.const,
                self.solver,
                rider_mass=self.mass,
                rider_x=key,
                initial=self.last_shape,
            )
            self.cache[key] = self.last_shape
        return self.cache[key]


def local_acceleration(
    x: float,
    v: float,
    shape: CableShape,
    cable: CableParams,
    const: PhysicalConstants,
    mass: float,
    mu: float,
    kd: float,
    ce: float,
    xb_ratio: float,
    FB: float,
) -> float:
    yx = slope(shape, x)
    kappa = curvature(shape, x)
    denom = np.sqrt(1.0 + yx * yx)
    sin_alpha = -yx / denom
    cos_alpha = 1.0 / denom
    normal = max(0.0, mass * const.g * cos_alpha + mass * v * v * kappa)
    brake = FB if x >= xb_ratio * cable.W else 0.0
    force = mass * const.g * sin_alpha - mu * normal - kd * v * abs(v) - ce * v - brake
    return float(force / mass)


def simulate_motion(
    cable: CableParams,
    const: PhysicalConstants,
    solver: SolverParams,
    mass: float,
    xb_ratio: float,
    FB: float,
    resistance: ResistanceParams,
    v0: float = 0.05,
    v_limit: float = 3.5,
    a_safe: float = 3.0,
    mu: float | None = None,
    kd: float | None = None,
    ce: float | None = None,
) -> SimulationResult:
    mu = resistance.mu_default if mu is None else mu
    kd = resistance.kd_default if kd is None else kd
    ce = resistance.ce_default if ce is None else ce
    cache = ShapeCache(cable, const, solver, mass)

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        x = float(np.clip(z[0], 0.0, cable.W))
        v = max(float(z[1]), 0.0)
        shape = cache.get(x)
        yx = slope(shape, x)
        dxdt = v / np.sqrt(1.0 + yx * yx)
        dvdt = local_acceleration(x, v, shape, cable, const, mass, mu, kd, ce, xb_ratio, FB)
        return np.array([dxdt, dvdt])

    def terminal_event(_t: float, z: np.ndarray) -> float:
        return float(z[0] - cable.W)

    terminal_event.terminal = True
    terminal_event.direction = 1

    def stall_event(_t: float, z: np.ndarray) -> float:
        x = float(np.clip(z[0], 0.0, cable.W))
        if x < 1.0 or x >= cable.W - 1e-6:
            return 1.0
        return float(z[1] - 1e-3)

    stall_event.terminal = True
    stall_event.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, 1000.0),
        np.array([0.0, v0]),
        rtol=solver.rtol,
        atol=solver.atol,
        max_step=solver.max_step,
        events=[terminal_event, stall_event],
    )
    x = np.clip(sol.y[0], 0.0, cable.W)
    v = np.maximum(sol.y[1], 0.0)
    acc = np.empty_like(v)
    T_max = 0.0
    sag_max_value = 0.0
    for i, (xi, vi) in enumerate(zip(x, v)):
        shape = cache.get(float(xi))
        acc[i] = local_acceleration(float(xi), float(vi), shape, cable, const, mass, mu, kd, ce, xb_ratio, FB)
        T_max = max(T_max, shape.T_max)
        sag_max_value = max(sag_max_value, max_sag(shape, cable))

    reached = bool(sol.t_events[0].size)
    stalled = (bool(sol.t_events[1].size) and not reached) or (not reached and x[-1] < cable.W and v[-1] <= 1e-2)
    imax = int(np.argmax(v)) if v.size else 0
    v_terminal = float(v[-1]) if reached else float("nan")
    max_decel = float(max(0.0, -np.min(acc))) if acc.size else 0.0
    max_accel = float(np.max(acc)) if acc.size else 0.0
    feasible = reached and not stalled and v_terminal <= v_limit and max_decel <= a_safe
    return SimulationResult(
        t=sol.t,
        x=x,
        v=v,
        accel=acc,
        reached_terminal=reached,
        stalled=stalled,
        v_max=float(np.max(v)) if v.size else 0.0,
        x_vmax=float(x[imax]) if x.size else 0.0,
        v_terminal=v_terminal,
        travel_time=float(sol.t[-1]) if sol.t.size else float("nan"),
        max_accel=max_accel,
        max_decel=max_decel,
        T_max=float(T_max),
        sag_max=float(sag_max_value),
        feasible=feasible,
        message=str(sol.message),
    )
