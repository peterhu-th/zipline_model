from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from config.params import Params, cable_length_from_eta
from core.braking import check_braking_feasibility
from core.dynamics import simulate_motion


def run(params: Params, limit_per_axis: int | None = None) -> pd.DataFrame:
    H_grid = params.grid.H_grid if limit_per_axis is None else params.grid.H_grid[:limit_per_axis]
    eta_grid = params.grid.eta_grid if limit_per_axis is None else params.grid.eta_grid[:limit_per_axis]
    xb_grid = params.brake.xb_ratio_grid if limit_per_axis is None else params.brake.xb_ratio_grid[:limit_per_axis]
    FB_grid = params.brake.FB_grid if limit_per_axis is None else params.brake.FB_grid[:limit_per_axis]
    rows = []
    for H in H_grid:
        for eta in eta_grid:
            L = cable_length_from_eta(params.cable.W, float(H), float(eta))
            cable = replace(params.cable, H=float(H), L=L)
            for xb_ratio in xb_grid:
                for FB in FB_grid:
                    sims = []
                    feasible = True
                    for mass in params.rider.mass_grid:
                        sim = simulate_motion(
                            cable,
                            params.const,
                            params.solver,
                            mass=mass,
                            xb_ratio=float(xb_ratio),
                            FB=float(FB),
                            resistance=params.resistance,
                            v0=params.rider.v0,
                            v_limit=params.rider.v_limit,
                            a_safe=params.brake.a_safe,
                        )
                        sims.append(sim)
                        feasible = feasible and check_braking_feasibility(
                            sim,
                            params.rider.v_limit,
                            params.brake.a_safe,
                            cable.T_allow,
                            cable.delta_allow,
                        )
                    speeds = [cable.W / s.travel_time for s in sims if s.reached_terminal]
                    avg_speed = float(np.mean(speeds)) if speeds else np.nan
                    rows.append(
                        {
                            "H": float(H),
                            "eta": float(eta),
                            "L": float(L),
                            "xb_ratio": float(xb_ratio),
                            "FB": float(FB),
                            "feasible": bool(feasible),
                            "score_avg_speed": float(avg_speed) if np.isfinite(avg_speed) else np.nan,
                            "v_T_30": sims[0].v_terminal,
                            "v_T_80": sims[-1].v_terminal,
                            "v_max_30": sims[0].v_max,
                            "v_max_80": sims[-1].v_max,
                            "max_decel": max(s.max_decel for s in sims),
                            "T_max": max(s.T_max for s in sims),
                            "sag_max": max(s.sag_max for s in sims),
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["feasible", "score_avg_speed"], ascending=[False, False])
    return df


if __name__ == "__main__":
    print(run(Params(), limit_per_axis=2).to_string(index=False))
