from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params, cable_length_from_eta
from core.braking import check_braking_feasibility
from core.dynamics import simulate_motion
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_global_feasible_region


def _mass_value(data: dict[float, object], mass: float, attr: str) -> float:
    """按质量读取仿真指标，避免依赖列表顺序"""
    sim = data.get(float(mass))
    return float(getattr(sim, attr)) if sim is not None else np.nan


def _jerk_metrics(sim) -> tuple[float, float]:
    """计算 jerk 和制动冲击指标"""
    if sim.t.size < 2:
        return 0.0, 0.0
    jerk = np.gradient(sim.accel, sim.t, edge_order=1)
    max_abs_jerk = float(np.max(np.abs(jerk))) if jerk.size else 0.0
    braking_jerk = jerk[sim.accel < 0.0]
    max_braking_jerk = float(np.max(np.abs(braking_jerk))) if braking_jerk.size else 0.0
    return max_abs_jerk, max_braking_jerk


def _scan(params: Params, H_grid, eta_grid, xb_grid, FB_grid, scan_stage: str) -> pd.DataFrame:
    """执行一轮参数扫描"""
    rows = []
    base_cable = recommended_cable(params)
    total = len(H_grid) * len(eta_grid) * len(xb_grid) * len(FB_grid)
    done = 0
    print(f"[扫描] {scan_stage} 阶段，参数组合数量：{total}")
    for H in H_grid:
        for eta in eta_grid:
            L = cable_length_from_eta(base_cable.W, float(H), float(eta))
            cable = replace(base_cable, H=float(H), L=L)
            for xb_ratio in xb_grid:
                for FB in FB_grid:
                    done += 1
                    print(
                        f"[进度] {scan_stage} {done}/{total}，"
                        f"H={float(H):.2f}，eta={float(eta):.4f}，xb/W={float(xb_ratio):.2f}，FB={float(FB):.0f}N"
                    )
                    sims: dict[float, object] = {}
                    feasible = True
                    for mass in params.rider.mass_grid:
                        sim = simulate_motion(
                            cable,
                            params.const,
                            params.solver,
                            mass=mass,
                            xb_ratio=float(xb_ratio),
                            FB=float(FB),
                            resistance=params.resistance,
                            v0=params.rider.v0,
                            v_limit=params.rider.v_limit,
                            a_safe=params.brake.a_safe,
                        )
                        sims[float(mass)] = sim
                        feasible = feasible and check_braking_feasibility(
                            sim,
                            params.rider.v_limit,
                            params.brake.a_safe,
                            cable.T_allow,
                            cable.delta_allow,
                        )
                    speeds = [s.avg_speed for s in sims.values() if s.reached_terminal and np.isfinite(s.avg_speed)]
                    avg_speed = float(np.mean(speeds)) if speeds else np.nan
                    jerk_values = [_jerk_metrics(s)[0] for s in sims.values()]
                    braking_jerk_values = [_jerk_metrics(s)[1] for s in sims.values()]
                    row = {
                        "scan_stage": scan_stage,
                        "H": float(H),
                        "eta": float(eta),
                        "L": float(L),
                        "xb_ratio": float(xb_ratio),
                        "FB": float(FB),
                        "feasible": bool(feasible),
                        "score_avg_speed": float(avg_speed) if np.isfinite(avg_speed) else np.nan,
                        "score_speed_s_over_t": float(avg_speed) if np.isfinite(avg_speed) else np.nan,
                        "v_T_30": _mass_value(sims, 30.0, "v_terminal"),
                        "v_T_80": _mass_value(sims, 80.0, "v_terminal"),
                        "v_max_30": _mass_value(sims, 30.0, "v_max"),
                        "v_max_80": _mass_value(sims, 80.0, "v_max"),
                        "max_decel": max(s.max_decel for s in sims.values()),
                        "max_abs_jerk": max(jerk_values),
                        "max_braking_jerk": max(braking_jerk_values),
                        "T_max": max(s.T_max for s in sims.values()),
                        "sag_max": max(s.sag_max for s in sims.values()),
                    }
                    for mass, sim in sims.items():
                        tag = int(mass) if float(mass).is_integer() else mass
                        row[f"v_avg_{tag}"] = sim.avg_speed
                    rows.append(row)
    return pd.DataFrame(rows)


def _local_grid(center: float, step: float, lower: float, upper: float, radius: int = 2) -> tuple[float, ...]:
    """围绕粗扫候选生成局部细化网格"""
    values = [center + step * offset for offset in range(-radius, radius + 1)]
    return tuple(sorted({float(np.clip(value, lower, upper)) for value in values}))


def _best_trajectory(params: Params, row: pd.Series) -> pd.DataFrame:
    """输出最优方案对应的完整轨迹"""
    cable = replace(recommended_cable(params), H=float(row["H"]), L=float(row["L"]))
    rows = []
    for mass in params.rider.mass_grid:
        sim = simulate_motion(
            cable,
            params.const,
            params.solver,
            mass=mass,
            xb_ratio=float(row["xb_ratio"]),
            FB=float(row["FB"]),
            resistance=params.resistance,
            v0=params.rider.v0,
            v_limit=params.rider.v_limit,
            a_safe=params.brake.a_safe,
        )
        jerk = np.gradient(sim.accel, sim.t, edge_order=1) if sim.t.size > 1 else np.zeros_like(sim.accel)
        for t, x, s, v, accel, j in zip(sim.t, sim.x, sim.s, sim.v, sim.accel, jerk):
            rows.append(
                {
                    "mass": float(mass),
                    "H": float(row["H"]),
                    "eta": float(row["eta"]),
                    "L": float(row["L"]),
                    "xb_ratio": float(row["xb_ratio"]),
                    "FB": float(row["FB"]),
                    "t": float(t),
                    "x": float(x),
                    "s": float(s),
                    "v": float(v),
                    "accel": float(accel),
                    "jerk": float(j),
                }
            )
    return pd.DataFrame(rows)


def run(params: Params, limit_per_axis: int | None = None) -> pd.DataFrame:
    """多目标优化"""
    H_grid = params.grid.H_grid if limit_per_axis is None else params.grid.H_grid[:limit_per_axis]
    eta_grid = params.grid.eta_grid if limit_per_axis is None else params.grid.eta_grid[:limit_per_axis]
    xb_grid = params.brake.xb_ratio_grid if limit_per_axis is None else params.brake.xb_ratio_grid[:limit_per_axis]
    FB_grid = params.brake.FB_grid if limit_per_axis is None else params.brake.FB_grid[:limit_per_axis]
    coarse = _scan(params, H_grid, eta_grid, xb_grid, FB_grid, "coarse")
    candidates = coarse.sort_values(["feasible", "score_avg_speed"], ascending=[False, False]).head(1 if limit_per_axis else 3)
    radius = 0 if limit_per_axis else 2
    refined_frames = []
    for _, row in candidates.iterrows():
        refined_frames.append(
            _scan(
                params,
                _local_grid(row["H"], 0.5, min(params.grid.H_grid), max(params.grid.H_grid), radius),
                _local_grid(row["eta"], 0.001, min(params.grid.eta_grid), max(params.grid.eta_grid), radius),
                _local_grid(row["xb_ratio"], 0.01, min(params.brake.xb_ratio_grid), max(params.brake.xb_ratio_grid), radius),
                _local_grid(row["FB"], 5.0, min(params.brake.FB_grid), max(params.brake.FB_grid), radius),
                "refined",
            )
        )
    df = pd.concat([coarse] + refined_frames, ignore_index=True) if refined_frames else coarse
    if not df.empty:
        df = df.sort_values(["feasible", "score_speed_s_over_t", "max_braking_jerk"], ascending=[False, False, True])
        df.attrs["extra_tables"] = {
            "exp08_best_trajectory.csv": _best_trajectory(params, df.iloc[0]),
        }
    return df


if __name__ == "__main__":
    run_experiment_outputs(
        "综合优化扫描",
        "exp08_global_optimization.csv",
        lambda: run(Params(), limit_per_axis=2),
        (plot_global_feasible_region,),
    )
