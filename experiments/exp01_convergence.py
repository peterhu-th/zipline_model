from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.energy_cable import CableShape, compute_tension, solve_cable_shape
from core.geometry import max_sag
from utils.experiment_runner import run_experiment_outputs


def _resample_initial(previous: CableShape | None, cable) -> CableShape | None:
    """把上一组 N 的索形重采样为当前 N 的热启动初值"""

    if previous is None:
        return None
    x = np.linspace(0.0, cable.W, cable.N + 1)
    y = np.interp(x, previous.x, previous.y)
    tension, lengths, rest_lengths = compute_tension(x, y, cable)
    return CableShape(
        x=x,
        y=y,
        tension=tension,
        lengths=lengths,
        rest_lengths=rest_lengths,
        success=previous.success,
        objective=previous.objective,
        message="由上一组 N 重采样得到",
    )


def run(params: Params) -> pd.DataFrame:
    """网格收敛性实验"""
    rows = []
    previous = None
    for N in (30, 50, 80, 100, 150, 200, 300, 400):
        cable = replace(params.cable, N=int(N))
        shape = solve_cable_shape(cable, params.const, params.solver, initial=_resample_initial(previous, cable))
        previous = shape
        rows.append(
            {
                "N": int(N),
                "success": shape.success,
                "arc_length": shape.arc_length,
                "T_max": shape.T_max,
                "sag_max": max_sag(shape, cable),
                "objective": shape.objective,
            }
        )
    df = pd.DataFrame(rows)
    ref = df.loc[df["N"] == 400].iloc[0]
    for name in ("arc_length", "T_max", "sag_max"):
        denom = ref[name] if abs(ref[name]) > 1e-12 else 1.0
        df[f"{name}_rel_error"] = (df[name] - ref[name]).abs() / abs(denom)
    return df


if __name__ == "__main__":
    run_experiment_outputs("网格收敛性实验", "exp01_convergence.csv", lambda: run(Params()))
