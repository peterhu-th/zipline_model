from __future__ import annotations
from dataclasses import replace
import pandas as pd
from config.params import Params
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


def run(params: Params) -> pd.DataFrame:
    """网格收敛性实验"""
    rows = []
    for N in (50, 100, 150, 200, 300, 400):
        cable = replace(params.cable, N=int(N))
        shape = solve_cable_shape(cable, params.const, params.solver)
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
    print(run(Params()).to_string(index=False))
