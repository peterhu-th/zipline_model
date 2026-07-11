from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params
from core.dynamics import simulate_motion, simulate_motion_fixed_shape
from core.energy_cable import CableShape, compute_tension, solve_cable_shape
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_motion_time_curves


MODEL_NAMES = {
    "moving": "准静态移动载荷模型",
    "fixed_empty": "固定空置索形模型",
    "fixed_loaded": "固定载人平均索形模型",
}


def _average_loaded_shape(params: Params, mass: float, empty_shape: CableShape, cable) -> CableShape:
    """计算固定载人平均索形"""
    loaded_shapes = [
        solve_cable_shape(
            cable,
            params.const,
            params.solver,
            rider_mass=mass,
            rider_x=ratio * cable.W,
        )
        for ratio in (0.25, 0.50, 0.75)
    ]
    avg_y = np.mean([shape.y for shape in loaded_shapes], axis=0)
    tension, lengths, rest_lengths = compute_tension(empty_shape.x, avg_y, cable)
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
    first = ["model", "mass", "terminal_s", "terminal_v", "terminal_a", "terminal_speed_safe", "failure_reason"]
    return {key: row[key] for key in first if key in row} | {key: value for key, value in row.items() if key not in first}


def _failure_reason(sim, params: Params) -> str:
    """按题目安全约束给出失败原因"""
    if not sim.reached_terminal:
        return "未到达终点"
    if sim.v_terminal > params.rider.v_limit:
        return "终点速度超限"
    if sim.max_decel > params.brake.a_safe:
        return "制动冲击超限"
    return "满足安全约束"


def _summary_row(sim, model: str, mass: float, params: Params) -> dict:
    """生成运动函数实验摘要行"""
    terminal_a = float(sim.accel[-1]) if sim.accel.size else np.nan
    terminal_s = float(sim.s[-1]) if sim.s.size else np.nan
    return {
        "model": model,
        "mass": float(mass),
        "terminal_s": terminal_s,
        "terminal_v": sim.v_terminal,
        "terminal_a": terminal_a,
        "terminal_speed_safe": bool(sim.reached_terminal and sim.v_terminal <= params.rider.v_limit),
        "failure_reason": _failure_reason(sim, params),
    }


def _simulate_for_mass(params: Params, mass: float):
    """生成某一质量下三种动力学模型的对比行"""
    rows = []
    cable = recommended_cable(params)
    empty_shape = solve_cable_shape(cable, params.const, params.solver)
    average_loaded_shape = _average_loaded_shape(params, mass, empty_shape, cable)
    moving = simulate_motion(
        cable,
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
        cable,
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
        cable,
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
    rows.append(_ordered_row(_summary_row(moving, MODEL_NAMES["moving"], mass, params)))
    rows.append(_ordered_row(_summary_row(fixed_empty, MODEL_NAMES["fixed_empty"], mass, params)))
    rows.append(_ordered_row(_summary_row(fixed_loaded, MODEL_NAMES["fixed_loaded"], mass, params)))
    return rows, moving


def _time_history_rows(mass: float, sim, sample_count: int = 101) -> list[dict[str, float | str]]:
    """按均匀时间采样提取准静态移动载荷模型的时程曲线数据"""

    if sim.t.size == 0:
        return []
    if sim.t.size == 1:
        sample_t = sim.t
    else:
        sample_t = np.linspace(float(sim.t[0]), float(sim.t[-1]), sample_count)
    return [
        {
            "model": MODEL_NAMES["moving"],
            "mass": mass,
            "t": float(t),
            "x": float(x),
            "s": float(s),
            "v": float(v),
            "a": float(accel),
        }
        for t, x, s, v, accel in zip(
            sample_t,
            np.interp(sample_t, sim.t, sim.x),
            np.interp(sample_t, sim.t, sim.s),
            np.interp(sample_t, sim.t, sim.v),
            np.interp(sample_t, sim.t, sim.accel),
        )
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
    df.attrs["extra_tables"] = {"exp05_motion_trajectory.csv": df.attrs["time_history"]}
    return df


if __name__ == "__main__":
    run_experiment_outputs(
        "动力学模型对比实验",
        "exp05_dynamic_model_compare.csv",
        lambda: run(Params()),
        (plot_motion_time_curves,),
    )
