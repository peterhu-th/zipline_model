from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from config.params import OUTPUT_DIR
from utils.io import ensure_output_dirs


def save_current_figure(name: str) -> Path:
    ensure_output_dirs()
    path = OUTPUT_DIR / "figures" / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path

