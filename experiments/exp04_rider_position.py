from __future__ import annotations
import pandas as pd
from config.params import Params
from core.energy_cable import solve_cable_shape
from core.geometry import interp_y, max_sag


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
    """载人位置对索形和张力的影响"""
    rows = []
    initial = None
    for ratio in params.grid.rider_pos_ratio_grid:
        rider_x = ratio * params.cable.W
        shape = solve_cable_shape(params.cable, params.const, params.solver, rider_mass=mass, rider_x=rider_x, initial=initial)
        initial = shape
        rows.append(
            {
                "mass": mass,
                "xi_ratio": ratio,
                "rider_x": rider_x,
                "rider_y": interp_y(shape, rider_x),
                "success": shape.success,
                "T_max": shape.T_max,
                "sag_max": max_sag(shape, params.cable),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))
