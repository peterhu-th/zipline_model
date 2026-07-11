from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.braking import scan_braking_region
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_braking_feasible_heatmap


def _analysis_table(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    """按质量汇总缓冲扫描无解原因"""
    rows = []
    for mass, part in df.groupby("mass"):
        reached = part[part["reached_terminal"]]
        min_terminal_speed = float(reached["v_terminal"].min()) if not reached.empty else float("nan")
        feasible_count = int(part["feasible"].sum())
        not_reached_count = int((~part["reached_terminal"]).sum())
        over_limit_count = int((reached["v_terminal"] > params.rider.v_limit).sum())
        if feasible_count:
            reason = "存在可行缓冲组合"
        elif reached.empty:
            reason = "扫描区内均未到达终点，缓冲过早或制动力过大"
        elif min_terminal_speed > params.rider.v_limit:
            reason = "可到达组合的最低终点速度仍超过限速"
        else:
            reason = "存在到达且低速组合，但受其他安全约束限制"
        rows.append(
            {
                "mass": float(mass),
                "feasible_count": feasible_count,
                "reached_count": int(part["reached_terminal"].sum()),
                "not_reached_count": not_reached_count,
                "terminal_speed_over_limit_count": over_limit_count,
                "min_terminal_speed": min_terminal_speed,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def run(params: Params, mass: float = 80.0):
    """缓冲可行域扫描"""
    cable = recommended_cable(params)
    masses = sorted({30.0, 50.0, float(mass)})
    df = pd.concat([scan_braking_region(params, mass=m, cable=cable) for m in masses], ignore_index=True)
    df.attrs["extra_tables"] = {"exp07_no_solution_analysis.csv": _analysis_table(df, params)}
    return df


if __name__ == "__main__":
    run_experiment_outputs(
        "缓冲可行域扫描",
        "exp07_braking_feasible_region.csv",
        lambda: run(Params()),
        (plot_braking_feasible_heatmap,),
    )
