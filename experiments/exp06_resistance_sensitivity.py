from __future__ import annotations
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config.params import Params
from core.braking import result_row
from core.dynamics import simulate_motion
from utils.experiment_runner import run_experiment_outputs


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
    """阻力参数敏感性"""
    rows = []
    total = len(params.resistance.mu_grid) * len(params.resistance.kd_grid) * len(params.resistance.ce_grid)
    done = 0
    print(f"[扫描] 阻力参数组合数量：{total}")
    for mu in params.resistance.mu_grid:
        for kd in params.resistance.kd_grid:
            for ce in params.resistance.ce_grid:
                done += 1
                print(f"[进度] 阻力敏感性 {done}/{total}，mu={mu:.3f}，kd={kd:.3f}，ce={ce:.3f}")
                sim = simulate_motion(
                    params.cable,
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
                rows.append(result_row(sim, mass=mass, mu=mu, kd=kd, ce=ce))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_experiment_outputs("阻力参数敏感性实验", "exp06_resistance_sensitivity.csv", lambda: run(Params()))
