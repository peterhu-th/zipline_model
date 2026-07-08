from __future__ import annotations
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
from utils.experiment_runner import run_experiment_outputs


def _rel_error(value: float, reference: float) -> float:
    """相对误差，参考值接近 0 时返回绝对误差"""

    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return abs(value - reference) / denom


def _append_compare_row(
    rows: list[dict[str, float | bool | str]],
    group: str,
    scenario: str,
    reference_model: str,
    compare_model: str,
    reference_success: bool,
    compare_success: bool,
    reference_T: float,
    compare_T: float,
    reference_sag: float,
    compare_sag: float,
    mass: float | None = None,
    xi_ratio: float | None = None,
) -> None:
    """追加一行模型对比结果"""

    rows.append(
        {
            "comparison_group": group,
            "scenario": scenario,
            "reference_model": reference_model,
            "compare_model": compare_model,
            "mass": np.nan if mass is None else float(mass),
            "xi_ratio": np.nan if xi_ratio is None else float(xi_ratio),
            "reference_success": bool(reference_success),
            "compare_success": bool(compare_success),
            "reference_T_max": float(reference_T),
            "compare_T_max": float(compare_T),
            "T_max_rel_error": _rel_error(compare_T, reference_T),
            "reference_sag_max": float(reference_sag),
            "compare_sag_max": float(compare_sag),
            "sag_max_rel_error": _rel_error(compare_sag, reference_sag),
        }
    )


def _analytic_empty_metrics(params: Params) -> tuple[bool, float, float]:
    """计算空置解析悬链线指标"""

    analytic = solve_analytic_catenary(params.cable)
    _, analytic_T = analytic_tension(analytic, params.cable, params.const)
    xs = np.linspace(0.0, params.cable.W, 400)
    chord = params.cable.H * (1.0 - xs / params.cable.W)
    analytic_sag = float(np.max(chord - analytic.y(xs)))
    return analytic.success, analytic_T, analytic_sag


def _energy_metrics(params: Params, mass: float | None = None, xi_ratio: float | None = None):
    """计算能量模型指标"""

    rider_x = None if xi_ratio is None else xi_ratio * params.cable.W
    shape = solve_cable_shape(params.cable, params.const, params.solver, rider_mass=mass, rider_x=rider_x)
    return shape.success, shape.T_max, max_sag(shape, params.cable)


def _segmented_metrics(params: Params, mass: float, xi_ratio: float):
    """计算分段悬链线指标"""

    segmented = solve_segmented_catenary_model(
        params.cable,
        rider_mass=mass,
        rider_x=xi_ratio * params.cable.W,
        const=params.const,
    )
    return segmented.success, segmented.T_max, segmented.sag_max


def run(params: Params) -> pd.DataFrame:
    """静力模型三组对比"""

    rows: list[dict[str, float | bool | str]] = []
    energy_success, energy_T, energy_sag = _energy_metrics(params)
    analytic_success, analytic_T, analytic_sag = _analytic_empty_metrics(params)
    _append_compare_row(
        rows,
        "空置状态",
        "空置钢缆",
        "能量最小化模型",
        "解析悬链线模型",
        energy_success,
        analytic_success,
        energy_T,
        analytic_T,
        energy_sag,
        analytic_sag,
    )

    normal_mass = float(params.rider.mass_grid[len(params.rider.mass_grid) // 2])
    for ratio in (0.25, 0.50, 0.75):
        energy_success, energy_T, energy_sag = _energy_metrics(params, normal_mass, ratio)
        segmented_success, segmented_T, segmented_sag = _segmented_metrics(params, normal_mass, ratio)
        _append_compare_row(
            rows,
            "载人状态",
            f"常规体重，位置 x/W={ratio:.2f}",
            "能量最小化模型",
            "分段悬链线模型",
            energy_success,
            segmented_success,
            energy_T,
            segmented_T,
            energy_sag,
            segmented_sag,
            normal_mass,
            ratio,
        )

    heavy_mass = float(max(params.rider.mass_grid))
    for ratio in (0.50, 0.75):
        energy_success, energy_T, energy_sag = _energy_metrics(params, heavy_mass, ratio)
        segmented_success, segmented_T, segmented_sag = _segmented_metrics(params, heavy_mass, ratio)
        _append_compare_row(
            rows,
            "大体重或跨中载荷",
            f"大体重载荷，位置 x/W={ratio:.2f}",
            "能量最小化模型",
            "分段悬链线模型",
            energy_success,
            segmented_success,
            energy_T,
            segmented_T,
            energy_sag,
            segmented_sag,
            heavy_mass,
            ratio,
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs("静力模型对比实验", "exp02_static_model_compare.csv", lambda: run(Params()))
