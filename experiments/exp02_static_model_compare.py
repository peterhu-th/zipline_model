from __future__ import annotations
import numpy as np
import pandas as pd
from config.params import Params
from core.baseline_models import (
    analytic_tension,
    solve_analytic_catenary,
    solve_segmented_catenary_model,
    solve_small_extension_model,
)
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


def _rel_error(value: float, reference: float) -> float:
    """相对误差，参考值接近 0 时返回绝对误差"""
    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return abs(value - reference) / denom


def run(params: Params) -> pd.DataFrame:
    """静力模型平衡对比"""
    rows = []
    analytic = solve_analytic_catenary(params.cable)
    _, analytic_T = analytic_tension(analytic, params.cable, params.const)
    xs = np.linspace(0.0, params.cable.W, 400)
    chord = params.cable.H * (1.0 - xs / params.cable.W)
    analytic_sag = float(np.max(chord - analytic.y(xs)))

    shape = solve_cable_shape(params.cable, params.const, params.solver)
    energy_sag = max_sag(shape, params.cable)
    small = solve_small_extension_model(params.cable, params.const)
    rows.extend(
        [
            {
                "state": "empty",
                "model": "analytic_catenary",
                "xi_ratio": np.nan,
                "mass": np.nan,
                "success": analytic.success,
                "T_max": analytic_T,
                "sag_max": analytic_sag,
                "T_max_rel_error": _rel_error(analytic_T, shape.T_max),
                "sag_max_rel_error": _rel_error(analytic_sag, energy_sag),
            },
            {
                "state": "empty",
                "model": "small_extension_catenary",
                "xi_ratio": np.nan,
                "mass": np.nan,
                "success": small.success,
                "T_max": small.T_max,
                "sag_max": small.sag_max,
                "T_max_rel_error": _rel_error(small.T_max, shape.T_max),
                "sag_max_rel_error": _rel_error(small.sag_max, energy_sag),
            },
            {
                "state": "empty",
                "model": "energy_minimization",
                "xi_ratio": np.nan,
                "mass": np.nan,
                "success": shape.success,
                "T_max": shape.T_max,
                "sag_max": energy_sag,
                "T_max_rel_error": 0.0,
                "sag_max_rel_error": 0.0,
            },
        ]
    )
    for ratio in (0.25, 0.50, 0.75):
        rider_x = ratio * params.cable.W
        energy_loaded = solve_cable_shape(
            params.cable,
            params.const,
            params.solver,
            rider_mass=max(params.rider.mass_grid),
            rider_x=rider_x,
        )
        segmented = solve_segmented_catenary_model(
            params.cable,
            rider_mass=max(params.rider.mass_grid),
            rider_x=rider_x,
            const=params.const,
        )
        energy_loaded_sag = max_sag(energy_loaded, params.cable)
        rows.extend(
            [
                {
                    "state": "loaded",
                    "model": "energy_minimization",
                    "xi_ratio": ratio,
                    "mass": max(params.rider.mass_grid),
                    "success": energy_loaded.success,
                    "T_max": energy_loaded.T_max,
                    "sag_max": energy_loaded_sag,
                    "T_max_rel_error": 0.0,
                    "sag_max_rel_error": 0.0,
                },
                {
                    "state": "loaded",
                    "model": "segmented_catenary",
                    "xi_ratio": ratio,
                    "mass": max(params.rider.mass_grid),
                    "success": segmented.success,
                    "T_max": segmented.T_max,
                    "sag_max": segmented.sag_max,
                    "T_max_rel_error": _rel_error(segmented.T_max, energy_loaded.T_max),
                    "sag_max_rel_error": _rel_error(segmented.sag_max, energy_loaded_sag),
                },
            ]
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))
