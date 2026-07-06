from __future__ import annotations

import argparse

from config.params import Params
from experiments import (
    exp01_convergence,
    exp02_static_model_compare,
    exp03_height_tension,
    exp04_rider_position,
    exp05_dynamic_model_compare,
    exp07_braking_feasible_region,
    exp08_global_optimization,
)
from utils.io import write_table


EXPERIMENTS = {
    "convergence": exp01_convergence.run,
    "static_compare": exp02_static_model_compare.run,
    "height_tension": exp03_height_tension.run,
    "rider_position": exp04_rider_position.run,
    "dynamic": exp05_dynamic_model_compare.run,
    "braking": exp07_braking_feasible_region.run,
}


def run_quick(params: Params) -> None:
    tables = {
        "exp01_convergence.csv": exp01_convergence.run(params),
        "exp02_static_model_compare.csv": exp02_static_model_compare.run(params),
        "exp03_height_tension.csv": exp03_height_tension.run(params),
        "exp04_rider_position.csv": exp04_rider_position.run(params),
        "exp05_dynamic_model_compare.csv": exp05_dynamic_model_compare.run(params),
    }
    for name, df in tables.items():
        path = write_table(df, name)
        print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zipline cable modeling experiments.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="quick",
        choices=["quick", "global"] + sorted(EXPERIMENTS),
        help="Experiment to run. Use quick for a small end-to-end smoke run.",
    )
    parser.add_argument("--limit-per-axis", type=int, default=2, help="Limit for global optimization smoke runs.")
    args = parser.parse_args()
    params = Params()

    if args.experiment == "quick":
        run_quick(params)
    elif args.experiment == "global":
        df = exp08_global_optimization.run(params, limit_per_axis=args.limit_per_axis)
        path = write_table(df, "exp08_global_optimization.csv")
        print(f"wrote {path}")
    else:
        df = EXPERIMENTS[args.experiment](params)
        path = write_table(df, f"{args.experiment}.csv")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
