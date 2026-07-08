from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params
from core.braking import result_row
from core.dynamics import simulate_motion, simulate_motion_fixed_shape
from core.energy_cable import CableShape, compute_tension, solve_cable_shape
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_motion_time_curves


MODEL_NAMES = {
    "moving": "准静态移动载荷模型",
    "fixed_empty": "固定空置索形模型",
    "fixed_loaded": "固定载人平均索形模型",
}


def _average_loaded_shape(params: Params, mass: float, empty_shape: CableShape) -> CableShape:
    """计算固定载人平均索形"""

    loaded_shapes = [
        solve_cable_shape(
            params.cable,
            params.const,
            params.solver,
            rider_mass=mass,
            rider_x=ratio * params.cable.W,
        )
        for ratio in (0.25, 0.50, 0.75)
    ]
    avg_y = np.mean([shape.y for shape in loaded_shapes], axis=0)
    tension, lengths, rest_lengths = compute_tension(empty_shape.x, avg_y, params.cable)
    return CableShape(
        x=empty_shape.x.copy(),
        y=avg_y,
        tension=tension,
        lengths=lengths,
        rest_lengths=rest_lengths,
        success=all(shape.success for shape in loaded_shapes),
        objective=float(np.mean([shape.objective for shape in loaded_shapes])),
        message="固定载人平均索形",
    )


def _ordered_row(row: dict) -> dict:
    """把模型和质量放在表格最左侧"""

    first = ["model", "mass", "reached_terminal", "stalled", "v_terminal", "v_max", "x_vmax", "travel_time", "avg_speed"]
    return {key: row[key] for key in first if key in row} | {key: value for key, value in row.items() if key not in first}


def _simulate_for_mass(params: Params, mass: float):
    """生成某一质量下三种动力学模型的对比行"""

    rows = []
    empty_shape = solve_cable_shape(params.cable, params.const, params.solver)
    average_loaded_shape = _average_loaded_shape(params, mass, empty_shape)
    moving = simulate_motion(
        params.cable,
        params.const,
        params.solver,
        mass=mass,
        xb_ratio=1.1,
        FB=0.0,
        resistance=params.resistance,
        v0=params.rider.v0,
        v_limit=params.rider.v_limit,
        a_safe=params.brake.a_safe,
    )
    fixed_empty = simulate_motion_fixed_shape(
        params.cable,
        params.const,
        params.solver,
        mass=mass,
        xb_ratio=1.1,
        FB=0.0,
        resistance=params.resistance,
        shape=empty_shape,
        v0=params.rider.v0,
        v_limit=params.rider.v_limit,
        a_safe=params.brake.a_safe,
    )
    fixed_loaded = simulate_motion_fixed_shape(
        params.cable,
        params.const,
        params.solver,
        mass=mass,
        xb_ratio=1.1,
        FB=0.0,
        resistance=params.resistance,
        shape=average_loaded_shape,
        v0=params.rider.v0,
        v_limit=params.rider.v_limit,
        a_safe=params.brake.a_safe,
    )
    rows.append(_ordered_row(result_row(moving, model=MODEL_NAMES["moving"], mass=mass, xb_ratio=1.1, FB=0.0)))
    rows.append(_ordered_row(result_row(fixed_empty, model=MODEL_NAMES["fixed_empty"], mass=mass, xb_ratio=1.1, FB=0.0)))
    rows.append(_ordered_row(result_row(fixed_loaded, model=MODEL_NAMES["fixed_loaded"], mass=mass, xb_ratio=1.1, FB=0.0)))
    return rows, moving


def _time_history_rows(mass: float, sim) -> list[dict[str, float | str]]:
    """提取准静态移动载荷模型的时程曲线数据"""

    return [
        {
            "model": MODEL_NAMES["moving"],
            "mass": mass,
            "t": float(t),
            "x": float(x),
            "s": float(s),
            "v": float(v),
            "accel": float(accel),
        }
        for t, x, s, v, accel in zip(sim.t, sim.x, sim.s, sim.v, sim.accel)
    ]


def run(params: Params, masses: tuple[float, ...] = (30.0, 50.0, 80.0)) -> pd.DataFrame:
    """动力学模型对比"""

    rows = []
    history_rows = []
    for mass in masses:
        print(f"[进度] 动力学模型对比，质量 {mass:g}kg")
        mass_rows, moving = _simulate_for_mass(params, mass)
        rows.extend(mass_rows)
        history_rows.extend(_time_history_rows(mass, moving))
    df = pd.DataFrame(rows)
    df.attrs["time_history"] = pd.DataFrame(history_rows)
    return df


if __name__ == "__main__":
    run_experiment_outputs(
        "动力学模型对比实验",
        "exp05_dynamic_model_compare.csv",
        lambda: run(Params()),
        (plot_motion_time_curves,),
    )
