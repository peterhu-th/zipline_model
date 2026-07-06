from __future__ import annotations

from dataclasses import replace

import pandas as pd

from config.params import Params
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


def run(params: Params) -> pd.DataFrame:
    rows = []
    for H in params.grid.H_grid:
        cable = replace(params.cable, H=float(H))
        shape = solve_cable_shape(cable, params.const, params.solver)
        rows.append({"H": float(H), "success": shape.success, "T_max": shape.T_max, "sag_max": max_sag(shape, cable)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))

