from __future__ import annotations
from dataclasses import asdict
import pandas as pd
from config.params import Params
from core.dynamics import SimulationResult, simulate_motion


def check_braking_feasibility(
    sim: SimulationResult,
    v_limit: float,
    a_safe: float,
    T_allow: float | None = None,
    delta_allow: float | None = None,
) -> bool:
    """验证动力学仿真是否满足边界约束"""
    ok = sim.reached_terminal and not sim.stalled
    ok = ok and sim.v_terminal <= v_limit
    ok = ok and sim.max_decel <= a_safe
    if T_allow is not None:
        ok = ok and sim.T_max <= T_allow
    if delta_allow is not None:
        ok = ok and sim.sag_max <= delta_allow
    return bool(ok)


def result_row(sim: SimulationResult, **context: float) -> dict[str, float | bool | str]:
    """构建数据行"""
    row = {k: v for k, v in asdict(sim).items() if k not in {"t", "x", "v", "accel"}}
    row.update(context)
    return row


def scan_braking_region(params: Params, mass: float, xb_ratios=None, FB_values=None) -> pd.DataFrame:
    """二维网格搜索，计算不同缓冲起始位置和制动力组合下的动力学响应"""
    xb_ratios = params.brake.xb_ratio_grid if xb_ratios is None else xb_ratios
    FB_values = params.brake.FB_grid if FB_values is None else FB_values
    rows = []
    for xb_ratio in xb_ratios:
        for FB in FB_values:
            sim = simulate_motion(
                params.cable,
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
            feasible = check_braking_feasibility(
                sim,
                params.rider.v_limit,
                params.brake.a_safe,
                params.cable.T_allow,
                params.cable.delta_allow,
            )
            rows.append(result_row(sim, mass=mass, xb_ratio=float(xb_ratio), FB=float(FB), feasible=feasible))
    return pd.DataFrame(rows)
