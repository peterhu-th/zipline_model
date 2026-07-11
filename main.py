from __future__ import annotations
import argparse
from config.params import Params
from experiments import (
    exp01_convergence,
    exp02_static_model_compare,
    exp03_height_tension,
    exp04_rider_position,
    exp05_dynamic_model_compare,
    exp06_resistance_sensitivity,
    exp07_braking_feasible_region,
    exp08_global_optimization,
)
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import (
    plot_H_Tmax_curve,
    plot_H_delta_max_curve,
    plot_motion_time_curves,
    plot_static_distribution_curves,
    plot_braking_feasible_heatmap,
    plot_global_feasible_region,
)


def experiment_specs(params: Params, limit_per_axis: int):
    """集中维护所有实验的生成逻辑，供 main 和 quick 复用"""

    return {
        "convergence": {
            "label": "网格收敛性实验",
            "table": "exp01_convergence.csv",
            "runner": lambda: exp01_convergence.run(params),
            "plotters": (),
        },
        "static_compare": {
            "label": "静力模型对比实验",
            "table": "exp02_static_model_compare.csv",
            "runner": lambda: exp02_static_model_compare.run(params),
            "plotters": (),
        },
        "height_tension": {
            "label": "高度差敏感性实验",
            "table": "exp03_height_tension.csv",
            "runner": lambda: exp03_height_tension.run(params),
            "plotters": (plot_H_Tmax_curve, plot_H_delta_max_curve),
        },
        "rider_position": {
            "label": "垂度和张力分布实验",
            "table": "exp04_static_distribution.csv",
            "runner": lambda: exp04_rider_position.run(params),
            "plotters": (plot_static_distribution_curves,),
        },
        "dynamic": {
            "label": "动力学模型对比实验",
            "table": "exp05_dynamic_model_compare.csv",
            "runner": lambda: exp05_dynamic_model_compare.run(params),
            "plotters": (plot_motion_time_curves,),
        },
        "resistance": {
            "label": "阻力参数敏感性实验",
            "table": "exp06_resistance_sensitivity.csv",
            "runner": lambda: exp06_resistance_sensitivity.run(params),
            "plotters": (),
        },
        "braking": {
            "label": "缓冲可行域扫描",
            "table": "exp07_braking_feasible_region.csv",
            "runner": lambda: exp07_braking_feasible_region.run(params),
            "plotters": (plot_braking_feasible_heatmap,),
        },
        "global_optimization": {
            "label": "综合优化扫描",
            "table": "exp08_global_optimization.csv",
            "runner": lambda: exp08_global_optimization.run(params, limit_per_axis=limit_per_axis),
            "plotters": (plot_global_feasible_region,),
        },
    }


def run_spec(spec: dict) -> None:
    """按统一规则运行一个实验并生成交付物"""

    run_experiment_outputs(spec["label"], spec["table"], spec["runner"], spec["plotters"])


def run_quick(specs: dict) -> None:
    """快速批量实验：运行前六个基础实验"""

    print("[开始] 快速批量实验")
    for key in ("convergence", "static_compare", "height_tension", "rider_position", "dynamic", "resistance"):
        run_spec(specs[key])
    print("[结束] 快速批量实验")


def main() -> None:
    params = Params()
    parser = argparse.ArgumentParser(description="运行钢缆溜索建模实验。")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="quick",
        choices=["quick", "global"] + sorted(experiment_specs(params, 2)),
        help="要运行的实验。quick 表示基础批量实验，global 是 global_optimization 的别名。",
    )
    parser.add_argument("--limit-per-axis", type=int, default=2, help="综合优化烟测时每个轴保留的网格数量。")
    args = parser.parse_args()
    specs = experiment_specs(params, args.limit_per_axis)

    if args.experiment == "quick":
        run_quick(specs)
    elif args.experiment == "global":
        run_spec(specs["global_optimization"])
    else:
        run_spec(specs[args.experiment])


if __name__ == "__main__":
    main()
