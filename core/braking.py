from __future__ import annotations
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


def scan_braking_region(params: Params, mass: float, xb_ratios=None, FB_values=None, cable=None) -> pd.DataFrame:
    """二维网格搜索，计算不同缓冲起始位置和制动力组合下的动力学响应"""
    xb_ratios = params.brake.xb_ratio_grid if xb_ratios is None else xb_ratios
    FB_values = params.brake.FB_grid if FB_values is None else FB_values
    cable = params.cable if cable is None else cable
    rows = []
    total = len(xb_ratios) * len(FB_values)
    done = 0
    print(f"[扫描] 质量 {mass:g}kg，缓冲组合数量：{total}")
    for xb_ratio in xb_ratios:
        for FB in FB_values:
            done += 1
            print(f"[进度] 质量 {mass:g}kg，{done}/{total}，xb/W={float(xb_ratio):.2f}，FB={float(FB):.0f}N")
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
            feasible = sim.reached_terminal and check_braking_feasibility(
                sim,
                params.rider.v_limit,
                params.brake.a_safe,
                cable.T_allow,
                cable.delta_allow,
            )
            rows.append(
                {
                    "mass": mass,
                    "xb_ratio": float(xb_ratio),
                    "FB": float(FB),
                    "reached_terminal": sim.reached_terminal,
                    "v_terminal": sim.v_terminal,
                    "v_max": sim.v_max,
                    "travel_time": sim.travel_time,
                    "arc_length": sim.arc_length,
                    "avg_speed": sim.avg_speed,
                    "feasible": bool(feasible),
                    "message": sim.message,
                }
            )
    return pd.DataFrame(rows)
