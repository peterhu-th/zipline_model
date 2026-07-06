from __future__ import annotations

from config.params import Params
from core.braking import scan_braking_region


def run(params: Params, mass: float = 80.0):
    return scan_braking_region(params, mass=mass)


if __name__ == "__main__":
    print(run(Params()).head().to_string(index=False))

