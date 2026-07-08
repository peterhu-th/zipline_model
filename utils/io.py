from __future__ import annotations
from pathlib import Path
import pandas as pd
from config.params import OUTPUT_DIR


def ensure_output_dirs() -> None:
    for name in ("figures", "tables", "logs"):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str) -> Path:
    ensure_output_dirs()
    path = OUTPUT_DIR / "tables" / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
