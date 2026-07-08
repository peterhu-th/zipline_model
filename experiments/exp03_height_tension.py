from __future__ import annotations
import pandas as pd
from config.params import Params
from core.sensitivity import tension_sensitivity_H


def run(params: Params) -> pd.DataFrame:
    """高度差对张力和垂度影响"""
    rows = []
    for H in params.grid.H_grid:
        rows.append(tension_sensitivity_H(params, H=float(H)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))
