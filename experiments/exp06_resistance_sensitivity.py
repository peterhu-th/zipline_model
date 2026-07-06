from __future__ import annotations

import pandas as pd

from config.params import Params
from core.braking import result_row
from core.dynamics import simulate_motion


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
    rows = []
    for mu in params.resistance.mu_grid:
        for kd in params.resistance.kd_grid:
            for ce in params.resistance.ce_grid:
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
    print(run(Params()).head().to_string(index=False))

