from __future__ import annotations
from collections.abc import Callable, Iterable
from pathlib import Path
import pandas as pd
from utils.io import write_table


Plotter = Callable[[pd.DataFrame], Path | Iterable[Path]]


def run_experiment_outputs(
    label: str,
    table_name: str,
    runner: Callable[[], pd.DataFrame],
    plotters: Iterable[Plotter] | None = None,
) -> pd.DataFrame:
    """统一执行实验、写表、绘图和打印运行反馈"""

    print(f"[开始] {label}")
    df = runner()
    print(f"[完成] {label}，结果行数：{len(df)}")
    table_path = write_table(df, table_name)
    print(f"[写入表格] {table_path}")

    for plotter in plotters or ():
        result = plotter(df)
        figure_paths = [result] if isinstance(result, Path) else list(result)
        for figure_path in figure_paths:
            print(f"[写入图表] {figure_path}")
    return df
