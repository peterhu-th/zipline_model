from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params
from core.energy_cable import solve_cable_shape
from core.geometry import interp_y, sag
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_static_distribution_curves


def _sample_tension(shape, xs: np.ndarray) -> np.ndarray:
    """把单元张力插值到采样位置"""
    centers = (shape.x[:-1] + shape.x[1:]) / 2.0
    return np.interp(xs, centers, shape.tension, left=shape.tension[0], right=shape.tension[-1])


def _shape_rows(state: str, shape, cable, mass: float | None = None, xi_ratio: float | None = None) -> list[dict]:
    """生成任意位置垂度和张力分布表"""
    xs = np.linspace(0.0, cable.W, 121)
    ys = np.interp(xs, shape.x, shape.y)
    chord = cable.H * (1.0 - xs / cable.W)
    tensions = _sample_tension(shape, xs)
    return [
        {
            "state": state,
            "mass": np.nan if mass is None else float(mass),
            "xi_ratio": np.nan if xi_ratio is None else float(xi_ratio),
            "x": float(x),
            "y": float(y),
            "sag": float(delta),
            "tension": float(tension),
        }
        for x, y, delta, tension in zip(xs, ys, chord - ys, tensions)
    ]


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
    """空置/载人条件下任意位置垂度和张力分布"""
    cable = recommended_cable(params)
    rows = []
    empty_shape = solve_cable_shape(cable, params.const, params.solver)
    rows.extend(_shape_rows("空置状态", empty_shape, cable))
    for ratio in (0.25, 0.50, 0.75):
        rider_x = ratio * cable.W
        loaded_shape = solve_cable_shape(cable, params.const, params.solver, rider_mass=mass, rider_x=rider_x)
        rows.extend(_shape_rows(f"载人状态 x/W={ratio:.2f}", loaded_shape, cable, mass=mass, xi_ratio=ratio))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs(
        "垂度和张力分布实验",
        "exp04_static_distribution.csv",
        lambda: run(Params()),
        (plot_static_distribution_curves,),
    )
