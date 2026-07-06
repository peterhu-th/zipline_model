from __future__ import annotations

import pandas as pd

from config.params import Params
from core.braking import result_row
from core.dynamics import simulate_motion


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
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
    )
    return pd.DataFrame([result_row(sim, model="quasi_static_moving_load", mass=mass, xb_ratio=1.1, FB=0.0)])


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))

