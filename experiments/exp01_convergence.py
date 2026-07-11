from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from experiments.common import convergence_table
from utils.experiment_runner import run_experiment_outputs


def run(params: Params) -> pd.DataFrame:
    """微元数收敛性实验：比较张力和垂度误差"""

    df = convergence_table(params)
    return df


if __name__ == "__main__":
    run_experiment_outputs("网格收敛性实验", "exp01_convergence.csv", lambda: run(Params()))
