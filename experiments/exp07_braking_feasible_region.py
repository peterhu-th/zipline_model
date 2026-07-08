from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.braking import scan_braking_region
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_braking_feasible_heatmap


def run(params: Params, mass: float = 80.0):
    """缓冲可行域扫描"""
    masses = sorted({30.0, 50.0, float(mass)})
    return pd.concat([scan_braking_region(params, mass=m) for m in masses], ignore_index=True)


if __name__ == "__main__":
    run_experiment_outputs(
        "缓冲可行域扫描",
        "exp07_braking_feasible_region.csv",
        lambda: run(Params()),
        (plot_braking_feasible_heatmap,),
    )
