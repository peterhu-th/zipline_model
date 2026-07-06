from __future__ import annotations

import numpy as np
import pandas as pd

from config.params import Params
from core.baseline_models import analytic_tension, solve_analytic_catenary
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


def run(params: Params) -> pd.DataFrame:
    analytic = solve_analytic_catenary(params.cable)
    _, analytic_T = analytic_tension(analytic, params.cable, params.const)
    xs = np.linspace(0.0, params.cable.W, 400)
    chord = params.cable.H * (1.0 - xs / params.cable.W)
    analytic_sag = float(np.max(chord - analytic.y(xs)))

    shape = solve_cable_shape(params.cable, params.const, params.solver)
    return pd.DataFrame(
        [
            {
                "model": "analytic_catenary",
                "success": analytic.success,
                "T_max": analytic_T,
                "sag_max": analytic_sag,
            },
            {
                "model": "energy_minimization",
                "success": shape.success,
                "T_max": shape.T_max,
                "sag_max": max_sag(shape, params.cable),
            },
        ]
    )


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))

