from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from config.params import Params, cable_length_from_eta
from core.braking import check_braking_feasibility
from core.dynamics import simulate_motion


def _mass_value(data: dict[float, object], mass: float, attr: str) -> float:
    """按质量读取仿真指标，避免依赖列表顺序"""
    sim = data.get(float(mass))
    return float(getattr(sim, attr)) if sim is not None else np.nan


def _scan(params: Params, H_grid, eta_grid, xb_grid, FB_grid, scan_stage: str) -> pd.DataFrame:
    """执行一轮参数扫描"""
    rows = []
    for H in H_grid:
        for eta in eta_grid:
            L = cable_length_from_eta(params.cable.W, float(H), float(eta))
            cable = replace(params.cable, H=float(H), L=L)
            for xb_ratio in xb_grid:
                for FB in FB_grid:
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
                    row = {
                        "scan_stage": scan_stage,
                        "H": float(H),
                        "eta": float(eta),
                        "L": float(L),
                        "xb_ratio": float(xb_ratio),
                        "FB": float(FB),
                        "feasible": bool(feasible),
                        "score_avg_speed": float(avg_speed) if np.isfinite(avg_speed) else np.nan,
                        "v_T_30": _mass_value(sims, 30.0, "v_terminal"),
                        "v_T_80": _mass_value(sims, 80.0, "v_terminal"),
                        "v_max_30": _mass_value(sims, 30.0, "v_max"),
                        "v_max_80": _mass_value(sims, 80.0, "v_max"),
                        "max_decel": max(s.max_decel for s in sims.values()),
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
        df = df.sort_values(["feasible", "score_avg_speed"], ascending=[False, False])
    return df


if __name__ == "__main__":
    print(run(Params(), limit_per_axis=2).to_string(index=False))
