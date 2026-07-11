from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.dynamics import simulate_motion
from experiments.common import recommended_cable
from utils.experiment_runner import run_experiment_outputs


def _row(sim, mass: float, mu: float, kd: float, ce: float, params: Params) -> dict:
    """阻力敏感性结果行，自变量放在前面"""
    terminal_speed_safe = bool(sim.reached_terminal and sim.v_terminal <= params.rider.v_limit)
    return {
        "mu": float(mu),
        "kd": float(kd),
        "ce": float(ce),
        "mass": float(mass),
        "v_terminal": sim.v_terminal,
        "v_max": sim.v_max,
        "reached_terminal": sim.reached_terminal,
        "terminal_speed_safe": terminal_speed_safe,
        "max_decel": sim.max_decel,
        "avg_speed": sim.avg_speed,
    }


def run(params: Params, masses: tuple[float, ...] = (30.0, 50.0, 80.0)) -> pd.DataFrame:
    """阻力参数敏感性"""
    rows = []
    cable = recommended_cable(params)
    total = len(masses) * len(params.resistance.mu_grid) * len(params.resistance.kd_grid) * len(params.resistance.ce_grid)
    done = 0
    print(f"[扫描] 阻力参数组合数量：{total}")
    for mass in masses:
        for mu in params.resistance.mu_grid:
            for kd in params.resistance.kd_grid:
                for ce in params.resistance.ce_grid:
                    done += 1
                    print(
                        f"[进度] 阻力敏感性 {done}/{total}，"
                        f"mass={mass:g}kg，mu={mu:.3f}，kd={kd:.3f}，ce={ce:.3f}"
                    )
                    sim = simulate_motion(
                        cable,
                        params.const,
                        params.solver,
                        mass=mass,
                        xb_ratio=1.1,
                        FB=0.0,
                        resistance=params.resistance,
                        v0=params.rider.v0,
                        v_limit=params.rider.v_limit,
                        a_safe=params.brake.a_safe,
                        mu=mu,
                        kd=kd,
                        ce=ce,
                    )
                    rows.append(_row(sim, mass, mu, kd, ce, params))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs("阻力参数敏感性实验", "exp06_resistance_sensitivity.csv", lambda: run(Params()))
