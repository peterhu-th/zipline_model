from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.sensitivity import tension_sensitivity_H
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_H_Tmax_curve, plot_H_delta_max_curve


def run(params: Params) -> pd.DataFrame:
    """高度差对张力和垂度影响"""
    rows = []
    base_params = replace(params, cable=recommended_cable(params))
    for H in params.grid.H_grid:
        result = tension_sensitivity_H(base_params, H=float(H))
        rows.append({key: value for key, value in result.items() if key not in {"dT_dH", "dSag_dH"}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs(
        "高度差敏感性实验",
        "exp03_height_tension.csv",
        lambda: run(Params()),
        (plot_H_Tmax_curve, plot_H_delta_max_curve),
    )
