from __future__ import annotations
import pandas as pd
from config.params import Params
from core.braking import scan_braking_region


def run(params: Params, mass: float = 80.0):
    """缓冲可行域扫描"""
    masses = sorted({30.0, 50.0, float(mass)})
    return pd.concat([scan_braking_region(params, mass=m) for m in masses], ignore_index=True)


if __name__ == "__main__":
    print(run(Params()).head().to_string(index=False))
