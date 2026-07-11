from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.params import CableParams, Params
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


N_CANDIDATES: tuple[int, ...] = (5, 8, 10, 12, 15, 100)
N_REFERENCE = 100
N_ERROR_TOL = 0.01


def _relative_error(value: float, reference: float) -> float:
    """计算相对误差，参考值接近 0 时按绝对误差处理"""
    denom = abs(reference) if abs(reference) > 1e-12 else 1.0
    return abs(value - reference) / denom


def convergence_table(params: Params) -> pd.DataFrame:
    """生成微元数收敛性表，并标记是否满足 1% 误差要求"""
    rows = []
    for N in N_CANDIDATES:
        cable = replace(params.cable, N=int(N))
        shape = solve_cable_shape(cable, params.const, params.solver)
        rows.append(
            {
                "N": int(N),
                "T_max": shape.T_max,
                "sag_max": max_sag(shape, cable),
            }
        )
    df = pd.DataFrame(rows)
    ref = df.loc[df["N"] == N_REFERENCE].iloc[0]
    df["T_max_rel_error"] = df["T_max"].map(lambda value: _relative_error(value, ref["T_max"]))
    df["sag_max_rel_error"] = df["sag_max"].map(lambda value: _relative_error(value, ref["sag_max"]))
    df["within_1_percent"] = (df["T_max_rel_error"] <= N_ERROR_TOL) & (df["sag_max_rel_error"] <= N_ERROR_TOL)
    return df


@lru_cache(maxsize=16)
def recommended_N(params: Params) -> int:
    """返回同时满足张力和垂度 1% 误差的最小微元数"""
    df = convergence_table(params)
    ok = df[(df["N"] != N_REFERENCE) & df["within_1_percent"]]
    if ok.empty:
        return N_REFERENCE
    return int(ok.sort_values("N").iloc[0]["N"])


def recommended_cable(params: Params) -> CableParams:
    """使用收敛性实验推荐的微元数构造后续实验钢缆参数"""
    return replace(params.cable, N=recommended_N(params))
