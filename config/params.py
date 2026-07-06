from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class PhysicalConstants:
    g: float = 9.8
    rho_air: float = 1.225


@dataclass(frozen=True)
class CableParams:
    W: float = 600.0
    H: float = 15.0
    L: float = 604.0
    R: float = 1.2
    E: float = 2.0e11
    A_c: float = 2.5e-4
    N: int = 50
    fixed_x: bool = True
    T_allow: float = 2.0e6
    delta_allow: float = 80.0

    @property
    def EA(self) -> float:
        return self.E * self.A_c


@dataclass(frozen=True)
class RiderParams:
    mass_grid: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    v_limit: float = 3.5
    v0: float = 0.05


@dataclass(frozen=True)
class ResistanceParams:
    mu_default: float = 0.02
    kd_default: float = 0.10
    ce_default: float = 2.0
    mu_grid: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)
    kd_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    ce_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class BrakeParams:
    xb_ratio_grid: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.92, 0.94)
    FB_grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0, 501, 20))
    a_safe: float = 3.0


@dataclass(frozen=True)
class SolverParams:
    optimizer_method: str = "SLSQP"
    max_iter: int = 400
    ftol: float = 1e-8
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step: float = 1.0
    diff_H: float = 0.1
    cache_dx: float = 20.0


@dataclass(frozen=True)
class ExperimentGrid:
    H_grid: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0)
    eta_grid: tuple[float, ...] = (0.002, 0.004, 0.006, 0.008, 0.010, 0.015)
    N_grid: tuple[int, ...] = (30, 50, 80)
    rider_pos_ratio_grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0.1, 1.0, 0.1))


@dataclass(frozen=True)
class Params:
    const: PhysicalConstants = field(default_factory=PhysicalConstants)
    cable: CableParams = field(default_factory=CableParams)
    rider: RiderParams = field(default_factory=RiderParams)
    resistance: ResistanceParams = field(default_factory=ResistanceParams)
    brake: BrakeParams = field(default_factory=BrakeParams)
    solver: SolverParams = field(default_factory=SolverParams)
    grid: ExperimentGrid = field(default_factory=ExperimentGrid)


def cable_length_from_eta(W: float, H: float, eta: float) -> float:
    return float(np.hypot(W, H) + eta * W)


DEFAULT_PARAMS = Params()
