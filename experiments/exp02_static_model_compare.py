from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params
from core.baseline_models import analytic_tension, solve_analytic_catenary, solve_segmented_catenary_model
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs


def _rel_error(value: float, reference: float) -> float:
    """相对误差，参考值接近 0 时返回绝对误差"""

    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return abs(value - reference) / denom


def _empty_rows(params: Params) -> list[dict[str, float | str]]:
    """空置状态：解析悬链线与能量模型对比"""

    cable = recommended_cable(params)
    energy = solve_cable_shape(cable, params.const, params.solver)
    energy_sag = max_sag(energy, cable)
    analytic = solve_analytic_catenary(cable)
    _, analytic_T = analytic_tension(analytic, cable, params.const)
    xs = np.linspace(0.0, cable.W, 400)
    analytic_sag = float(np.max(cable.H * (1.0 - xs / cable.W) - analytic.y(xs)))
    return [
        {
            "case": "空置状态",
            "model": "Model A：解析悬链线",
            "T_max": analytic_T,
            "sag_max": analytic_sag,
            "T_max_rel_error": _rel_error(analytic_T, energy.T_max),
            "sag_max_rel_error": _rel_error(analytic_sag, energy_sag),
        },
        {
            "case": "空置状态",
            "model": "Model B：能量最小化模型",
            "T_max": energy.T_max,
            "sag_max": energy_sag,
            "T_max_rel_error": 0.0,
            "sag_max_rel_error": 0.0,
        },
    ]


def _loaded_rows(params: Params, mass: float, xi_ratio: float, case: str) -> list[dict[str, float | str]]:
    """载人状态：分段悬链线与能量模型对比"""

    cable = recommended_cable(params)
    rider_x = xi_ratio * cable.W
    energy = solve_cable_shape(cable, params.const, params.solver, rider_mass=mass, rider_x=rider_x)
    energy_sag = max_sag(energy, cable)
    segmented = solve_segmented_catenary_model(cable, rider_mass=mass, rider_x=rider_x, const=params.const)
    return [
        {
            "case": case,
            "model": "Model C：分段悬链线",
            "T_max": segmented.T_max,
            "sag_max": segmented.sag_max,
            "T_max_rel_error": _rel_error(segmented.T_max, energy.T_max),
            "sag_max_rel_error": _rel_error(segmented.sag_max, energy_sag),
        },
        {
            "case": case,
            "model": "Model B：能量最小化模型",
            "T_max": energy.T_max,
            "sag_max": energy_sag,
            "T_max_rel_error": 0.0,
            "sag_max_rel_error": 0.0,
        },
    ]


def run(params: Params) -> pd.DataFrame:
    """静力模型对比"""

    normal_mass = float(params.rider.mass_grid[len(params.rider.mass_grid) // 2])
    heavy_mass = float(max(params.rider.mass_grid))
    rows = []
    rows.extend(_empty_rows(params))
    rows.extend(_loaded_rows(params, normal_mass, 0.50, "载人状态：跨中常规体重"))
    rows.extend(_loaded_rows(params, heavy_mass, 0.50, "大体重或跨中载荷"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs("静力模型对比实验", "exp02_static_model_compare.csv", lambda: run(Params()))
