from __future__ import annotations
import numpy as np
import pandas as pd
from config.params import Params
from core.braking import result_row
from core.dynamics import simulate_motion, simulate_motion_fixed_shape
from core.energy_cable import CableShape, compute_tension, solve_cable_shape


def run(params: Params, mass: float = 80.0) -> pd.DataFrame:
    """动力学模型对比"""
    rows = []
    empty_shape = solve_cable_shape(params.cable, params.const, params.solver)
    loaded_shapes = [
        solve_cable_shape(
            params.cable,
            params.const,
            params.solver,
            rider_mass=mass,
            rider_x=ratio * params.cable.W,
        )
        for ratio in (0.25, 0.50, 0.75)
    ]
    avg_y = np.mean([shape.y for shape in loaded_shapes], axis=0)
    tension, lengths, rest_lengths = compute_tension(empty_shape.x, avg_y, params.cable)
    average_loaded_shape = CableShape(
        x=empty_shape.x.copy(),
        y=avg_y,
        tension=tension,
        lengths=lengths,
        rest_lengths=rest_lengths,
        success=all(shape.success for shape in loaded_shapes),
        objective=float(np.mean([shape.objective for shape in loaded_shapes])),
        message="固定载人平均索形",
    )
    moving = simulate_motion(
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
    fixed_empty = simulate_motion_fixed_shape(
        params.cable,
        params.const,
        params.solver,
        mass=mass,
        xb_ratio=1.1,
        FB=0.0,
        resistance=params.resistance,
        shape=empty_shape,
        v0=params.rider.v0,
        v_limit=params.rider.v_limit,
        a_safe=params.brake.a_safe,
    )
    fixed_loaded = simulate_motion_fixed_shape(
        params.cable,
        params.const,
        params.solver,
        mass=mass,
        xb_ratio=1.1,
        FB=0.0,
        resistance=params.resistance,
        shape=average_loaded_shape,
        v0=params.rider.v0,
        v_limit=params.rider.v_limit,
        a_safe=params.brake.a_safe,
    )
    rows.append(result_row(moving, model="quasi_static_moving_load", mass=mass, xb_ratio=1.1, FB=0.0))
    rows.append(result_row(fixed_empty, model="fixed_empty_shape", mass=mass, xb_ratio=1.1, FB=0.0))
    rows.append(result_row(fixed_loaded, model="fixed_average_loaded_shape", mass=mass, xb_ratio=1.1, FB=0.0))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run(Params()).to_string(index=False))
