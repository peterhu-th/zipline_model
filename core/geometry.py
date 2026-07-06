from __future__ import annotations

import numpy as np

from config.params import CableParams
from core.energy_cable import CableShape


def interp_y(shape: CableShape, x: float) -> float:
    return float(np.interp(np.clip(x, shape.x[0], shape.x[-1]), shape.x, shape.y))


def slope(shape: CableShape, x: float) -> float:
    x = float(np.clip(x, shape.x[0], shape.x[-1]))
    dydx = np.gradient(shape.y, shape.x, edge_order=1)
    return float(np.interp(x, shape.x, dydx))


def curvature(shape: CableShape, x: float) -> float:
    x = float(np.clip(x, shape.x[0], shape.x[-1]))
    dydx = np.gradient(shape.y, shape.x, edge_order=1)
    d2ydx2 = np.gradient(dydx, shape.x, edge_order=1)
    y1 = float(np.interp(x, shape.x, dydx))
    y2 = float(np.interp(x, shape.x, d2ydx2))
    return float(abs(y2) / max((1.0 + y1 * y1) ** 1.5, 1e-12))


def sag(shape: CableShape, cable: CableParams) -> np.ndarray:
    chord = cable.H * (1.0 - shape.x / cable.W)
    return chord - shape.y


def max_sag(shape: CableShape, cable: CableParams) -> float:
    return float(np.max(sag(shape, cable)))


def arc_length(shape: CableShape) -> float:
    return float(np.sum(np.hypot(np.diff(shape.x), np.diff(shape.y))))


def sample_shape(shape: CableShape, num: int = 200) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(shape.x[0], shape.x[-1], num)
    ys = np.interp(xs, shape.x, shape.y)
    return xs, ys

