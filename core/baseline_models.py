from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from config.params import CableParams, PhysicalConstants


@dataclass
class AnalyticCatenary:
    a: float
    c: float
    d: float
    success: bool

    def y(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.a * np.cosh((np.asarray(x) - self.c) / self.a) + self.d

    def dydx(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.sinh((np.asarray(x) - self.c) / self.a)


def solve_analytic_catenary(cable: CableParams) -> AnalyticCatenary:
    D = float(np.hypot(cable.W, cable.H))
    if cable.L <= D:
        # A taut straight cable is the limiting case; use a large a.
        a = 1e9
        c = cable.W / 2.0
        d = cable.H / 2.0 - a
        return AnalyticCatenary(a=a, c=c, d=d, success=False)

    # Stable inclined-span parameterization:
    # sqrt(L^2-H^2)/W = sinh(m)/m, a = W/(2m).
    target = np.sqrt(max(cable.L * cable.L - cable.H * cable.H, 0.0)) / cable.W

    def f(m: float) -> float:
        return np.sinh(m) / m - target

    upper = 1.0
    while f(upper) < 0.0 and upper < 100.0:
        upper *= 2.0
    try:
        m = brentq(f, 1e-10, upper, maxiter=200)
        a = cable.W / (2.0 * m)
        n = -np.arcsinh(cable.H / (2.0 * a * np.sinh(m)))
        p = m + n
        q = m - n
        c = a * q
        d = -a * np.cosh(p)
        return AnalyticCatenary(a=float(a), c=float(c), d=float(d), success=True)
    except (ValueError, FloatingPointError, OverflowError):
        a = 1e9
        c = cable.W / 2.0
        d = cable.H / 2.0 - a
        return AnalyticCatenary(a=a, c=c, d=d, success=False)


def analytic_tension(model: AnalyticCatenary, cable: CableParams, const: PhysicalConstants) -> tuple[float, float]:
    horizontal = cable.R * const.g * model.a
    xs = np.linspace(0.0, cable.W, 400)
    total = horizontal * np.sqrt(1.0 + np.asarray(model.dydx(xs)) ** 2)
    return float(horizontal), float(np.max(total))


def solve_small_extension_model(cable: CableParams) -> AnalyticCatenary:
    return solve_analytic_catenary(cable)


def solve_segmented_catenary_model(cable: CableParams, rider_mass: float, rider_x: float) -> dict[str, float]:
    # Kept as a lightweight comparison placeholder; the main model handles load rigorously.
    return {
        "rider_x": float(rider_x),
        "rider_mass": float(rider_mass),
        "note": "Use the energy minimization model for production calculations.",
    }
