from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from config.params import OUTPUT_DIR
from utils.io import ensure_output_dirs


def configure_chinese_font() -> None:
    """配置 Matplotlib 中文字体，避免坐标轴文字乱码"""

    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS")
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_current_figure(name: str) -> Path:
    ensure_output_dirs()
    path = OUTPUT_DIR / "figures" / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_H_Tmax_curve(df: pd.DataFrame, name: str = "exp03_H_Tmax_curve.png") -> Path:
    """绘制 H-Tmax 曲线"""

    configure_chinese_font()
    plt.figure(figsize=(6, 4))
    plt.plot(df["H"], df["T_max"], marker="o")
    plt.xlabel("高度差 H / m")
    plt.ylabel("最大张力 Tmax / N")
    plt.grid(True, alpha=0.3)
    return save_current_figure(name)


def plot_H_delta_max_curve(df: pd.DataFrame, name: str = "exp03_H_delta_max_curve.png") -> Path:
    """绘制 H-最大垂度曲线"""

    configure_chinese_font()
    y_col = "sag_max" if "sag_max" in df.columns else "delta_max"
    plt.figure(figsize=(6, 4))
    plt.plot(df["H"], df[y_col], marker="o")
    plt.xlabel("高度差 H / m")
    plt.ylabel("最大垂度 δmax / m")
    plt.grid(True, alpha=0.3)
    return save_current_figure(name)


def plot_speed_position_curve(df: pd.DataFrame, name: str = "exp05_speed_position_curve.png") -> Path:
    """绘制速度-位置曲线"""

    configure_chinese_font()
    plt.figure(figsize=(6, 4))
    if "model" in df.columns:
        for label, part in df.groupby("model"):
            plt.plot(part["x"], part["v"], label=str(label))
        plt.legend()
    else:
        plt.plot(df["x"], df["v"])
    plt.xlabel("水平位置 x / m")
    plt.ylabel("速度 v / (m/s)")
    plt.grid(True, alpha=0.3)
    return save_current_figure(name)


def plot_motion_time_curves(df: pd.DataFrame) -> list[Path]:
    """绘制不同体重下的 x-t、s-t、v-t、a-t 曲线"""

    history = df.attrs.get("time_history")
    if history is None or history.empty:
        return []
    specs = [
        ("x", "水平位置 x / m", "exp05_x_t_curve.png"),
        ("s", "下滑位移 s / m", "exp05_s_t_curve.png"),
        ("v", "速度 v / (m/s)", "exp05_v_t_curve.png"),
        ("accel", "切向加速度 a / (m/s^2)", "exp05_a_t_curve.png"),
    ]
    paths = []
    for column, ylabel, filename in specs:
        configure_chinese_font()
        plt.figure(figsize=(7, 4.5))
        for mass, part in history.groupby("mass"):
            plt.plot(part["t"], part[column], label=f"{mass:g} kg")
        plt.xlabel("时间 t / s")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend(title="体重")
        paths.append(save_current_figure(filename))
    return paths


def plot_braking_feasible_heatmap(df: pd.DataFrame, name: str = "exp07_braking_feasible_heatmap.png") -> Path:
    """绘制缓冲可行域热力图"""

    configure_chinese_font()
    data = df.copy()
    if "mass" in data.columns:
        data = data[data["mass"] == data["mass"].max()]
    pivot = data.pivot_table(index="FB", columns="xb_ratio", values="feasible", aggfunc="max").astype(float)
    plt.figure(figsize=(7, 4.5))
    plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="Greens")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{x:.0f}" for x in pivot.index])
    plt.xlabel("缓冲起始位置 xb/W")
    plt.ylabel("缓冲摩擦力 FB / N")
    plt.colorbar(label="可行性")
    return save_current_figure(name)


def plot_global_feasible_region(df: pd.DataFrame, name: str = "exp08_global_feasible_region.png") -> Path:
    """绘制综合优化可行域"""

    configure_chinese_font()
    data = df.copy()
    colors = data["feasible"].map({True: "tab:green", False: "tab:red"})
    sizes = 40 + 120 * data["score_avg_speed"].fillna(0.0) / max(float(data["score_avg_speed"].fillna(0.0).max()), 1e-12)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(data["H"], data["eta"], c=colors, s=sizes, alpha=0.75)
    plt.xlabel("高度差 H / m")
    plt.ylabel("余长比例 eta")
    plt.grid(True, alpha=0.3)
    return save_current_figure(name)
