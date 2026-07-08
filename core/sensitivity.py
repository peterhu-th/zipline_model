from __future__ import annotations
from dataclasses import replace
from config.params import Params
from core.energy_cable import solve_cable_shape
from core.geometry import max_sag


def tension_sensitivity_H(params: Params, H: float | None = None, delta_H: float | None = None) -> dict[str, float]:
    """中心差分计算偏导"""
    H = params.cable.H if H is None else H
    delta_H = params.solver.diff_H if delta_H is None else delta_H
    c_minus = replace(params.cable, H=H - delta_H)
    c_mid = replace(params.cable, H=H)
    c_plus = replace(params.cable, H=H + delta_H)

    s_minus = solve_cable_shape(c_minus, params.const, params.solver)
    s_mid = solve_cable_shape(c_mid, params.const, params.solver)
    s_plus = solve_cable_shape(c_plus, params.const, params.solver)

    dT_dH = (s_plus.T_max - s_minus.T_max) / (2.0 * delta_H)
    dSag_dH = (max_sag(s_plus, c_plus) - max_sag(s_minus, c_minus)) / (2.0 * delta_H)
    sag_mid = max_sag(s_mid, c_mid)
    return {
        "H": float(H),
        "T_max": s_mid.T_max,
        "sag_max": sag_mid,
        "dT_dH": float(dT_dH),
        "dSag_dH": float(dSag_dH),
        "S_T": float(H / s_mid.T_max * dT_dH) if s_mid.T_max else 0.0,
        "S_sag": float(H / sag_mid * dSag_dH) if sag_mid else 0.0,
    }
