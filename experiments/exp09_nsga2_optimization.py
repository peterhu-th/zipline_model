from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config.params import Params, cable_length_from_eta
from core.braking import check_braking_feasibility
from core.dynamics import simulate_motion
from experiments.common import recommended_cable
from experiments.exp08_global_optimization import _jerk_metrics
from utils.experiment_runner import run_experiment_outputs
from utils.plotting import plot_nsga2_pareto_front


VARIABLES = ("H", "eta", "xb_ratio", "FB")


@dataclass
class Individual:
    """NSGA-II 个体"""
    x: np.ndarray
    objectives: np.ndarray | None = None
    violation: float = np.inf
    row: dict | None = None
    rank: int = 0
    crowding: float = 0.0


def _bounds(params: Params) -> np.ndarray:
    """由现有实验网格给出连续优化变量边界"""
    return np.array(
        [
            [min(params.grid.H_grid), max(params.grid.H_grid)],
            [min(params.grid.eta_grid), max(params.grid.eta_grid)],
            [min(params.brake.xb_ratio_grid), max(params.brake.xb_ratio_grid)],
            [min(params.brake.FB_grid), max(params.brake.FB_grid)],
        ],
        dtype=float,
    )


def _terminal_error(sims: dict[float, object], params: Params) -> float:
    """计算终点速度接近限速的程度，未到达时给大惩罚"""
    values = []
    for sim in sims.values():
        if not sim.reached_terminal or not np.isfinite(sim.v_terminal):
            values.append(params.rider.v_limit)
        else:
            values.append(abs(sim.v_terminal - params.rider.v_limit))
    return float(np.mean(values)) if values else float("inf")


def _constraint_violation(sims: dict[float, object], params: Params, cable) -> float:
    """把安全约束违反量汇总为一个非负数，用于约束支配排序"""
    violation = 0.0
    for sim in sims.values():
        if not sim.reached_terminal:
            violation += 100.0
        else:
            violation += max(0.0, sim.v_terminal - params.rider.v_limit) / max(params.rider.v_limit, 1e-12)
        violation += max(0.0, sim.max_decel - params.brake.a_safe) / max(params.brake.a_safe, 1e-12)
        violation += max(0.0, sim.T_max - cable.T_allow) / max(cable.T_allow, 1e-12)
        violation += max(0.0, sim.sag_max - cable.delta_allow) / max(cable.delta_allow, 1e-12)
    return float(violation)


def _evaluate(ind: Individual, params: Params, generation: int) -> Individual:
    """运行一次多体重仿真并计算 NSGA-II 目标函数"""
    H, eta, xb_ratio, FB = map(float, ind.x)
    base_cable = recommended_cable(params)
    L = cable_length_from_eta(base_cable.W, H, eta)
    cable = replace(base_cable, H=H, L=L)
    sims = {}
    feasible = True
    for mass in params.rider.mass_grid:
        sim = simulate_motion(
            cable,
            params.const,
            params.solver,
            mass=float(mass),
            xb_ratio=xb_ratio,
            FB=FB,
            resistance=params.resistance,
            v0=params.rider.v0,
            v_limit=params.rider.v_limit,
            a_safe=params.brake.a_safe,
        )
        sims[float(mass)] = sim
        feasible = feasible and check_braking_feasibility(
            sim,
            params.rider.v_limit,
            params.brake.a_safe,
            cable.T_allow,
            cable.delta_allow,
        )

    reached_speeds = [sim.avg_speed for sim in sims.values() if sim.reached_terminal and np.isfinite(sim.avg_speed)]
    avg_speed = float(np.mean(reached_speeds)) if reached_speeds else 0.0
    jerk_values = [_jerk_metrics(sim)[1] for sim in sims.values()]
    terminal_error = _terminal_error(sims, params)
    max_braking_jerk = float(max(jerk_values)) if jerk_values else float("inf")
    violation = _constraint_violation(sims, params, cable)

    ind.objectives = np.array([-avg_speed, max_braking_jerk, terminal_error], dtype=float)
    ind.violation = violation
    ind.row = {
        "H": H,
        "eta": eta,
        "L": float(L),
        "xb_ratio": xb_ratio,
        "FB": FB,
        "feasible": bool(feasible and violation <= 1e-12),
        "constraint_violation": violation,
        "score_speed_s_over_t": avg_speed,
        "max_braking_jerk": max_braking_jerk,
        "terminal_speed_error": terminal_error,
        "reached_count": int(sum(sim.reached_terminal for sim in sims.values())),
        "v_terminal_max": float(max((sim.v_terminal for sim in sims.values() if sim.reached_terminal), default=np.nan)),
        "v_terminal_min": float(min((sim.v_terminal for sim in sims.values() if sim.reached_terminal), default=np.nan)),
    }
    return ind


def _dominates(a: Individual, b: Individual) -> bool:
    """约束支配关系：先比较约束违反量，再比较多目标 Pareto 支配"""
    if a.violation < b.violation - 1e-12:
        return True
    if a.violation > b.violation + 1e-12:
        return False
    assert a.objectives is not None and b.objectives is not None
    return bool(np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives))


def _fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    """快速非支配排序"""
    dominates: dict[int, list[int]] = {i: [] for i in range(len(population))}
    dominated_count = {i: 0 for i in range(len(population))}
    fronts: list[list[int]] = [[]]
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if _dominates(p, q):
                dominates[i].append(j)
            elif _dominates(q, p):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)

    current = 0
    while current < len(fronts) and fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    population[j].rank = current + 1
                    next_front.append(j)
        current += 1
        if next_front:
            fronts.append(next_front)
    return [[population[i] for i in front] for front in fronts]


def _assign_crowding(front: list[Individual]) -> None:
    """计算拥挤距离，保留 Pareto 前沿多样性"""
    if not front:
        return
    for ind in front:
        ind.crowding = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowding = float("inf")
        return
    objective_count = len(front[0].objectives)
    for k in range(objective_count):
        front.sort(key=lambda item: item.objectives[k])
        front[0].crowding = front[-1].crowding = float("inf")
        low = front[0].objectives[k]
        high = front[-1].objectives[k]
        denom = max(high - low, 1e-12)
        for i in range(1, len(front) - 1):
            front[i].crowding += float((front[i + 1].objectives[k] - front[i - 1].objectives[k]) / denom)


def _select_next(combined: list[Individual], size: int) -> list[Individual]:
    """按等级和拥挤距离选择下一代"""
    selected: list[Individual] = []
    for front in _fast_non_dominated_sort(combined):
        _assign_crowding(front)
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            front.sort(key=lambda item: item.crowding, reverse=True)
            selected.extend(front[: size - len(selected)])
            break
    return selected


def _tournament(population: list[Individual], rng: np.random.Generator) -> Individual:
    """二元锦标赛选择"""
    i, j = rng.choice(len(population), size=2, replace=False)
    a, b = population[int(i)], population[int(j)]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding >= b.crowding else b


def _make_child(
    parent_a: Individual,
    parent_b: Individual,
    bounds: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float,
) -> Individual:
    """用算术交叉和高斯变异生成子代"""
    alpha = rng.uniform(0.0, 1.0, size=len(VARIABLES))
    child = alpha * parent_a.x + (1.0 - alpha) * parent_b.x
    span = bounds[:, 1] - bounds[:, 0]
    mask = rng.random(len(VARIABLES)) < mutation_rate
    child[mask] += rng.normal(0.0, 0.08, size=int(mask.sum())) * span[mask]
    child = np.clip(child, bounds[:, 0], bounds[:, 1])
    return Individual(child)


def _initial_population(params: Params, population_size: int, rng: np.random.Generator) -> list[Individual]:
    """生成初始种群，混入现有网格中点以提高稳定性"""
    bounds = _bounds(params)
    seeds = [
        np.array([np.median(params.grid.H_grid), np.median(params.grid.eta_grid), np.median(params.brake.xb_ratio_grid), np.median(params.brake.FB_grid)]),
        np.array([max(params.grid.H_grid), max(params.grid.eta_grid), max(params.brake.xb_ratio_grid), min(params.brake.FB_grid)]),
        np.array([max(params.grid.H_grid), max(params.grid.eta_grid), max(params.brake.xb_ratio_grid), np.median(params.brake.FB_grid)]),
    ]
    population = [Individual(np.clip(seed.astype(float), bounds[:, 0], bounds[:, 1])) for seed in seeds[:population_size]]
    while len(population) < population_size:
        population.append(Individual(rng.uniform(bounds[:, 0], bounds[:, 1])))
    return population


def _rows(population: list[Individual]) -> pd.DataFrame:
    """把最终种群中的 Pareto 前沿转换为精简结果表"""
    rows = []
    for ind in population:
        if ind.rank != 0:
            continue
        row = dict(ind.row or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["feasible", "score_speed_s_over_t", "max_braking_jerk"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _population_rows(population: list[Individual]) -> pd.DataFrame:
    """生成绘图使用的最终种群数据，不写入附表"""

    rows = []
    for ind in population:
        row = dict(ind.row or {})
        row["is_pareto"] = ind.rank == 0
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _best_trajectory(params: Params, row: pd.Series) -> pd.DataFrame:
    """输出最优方案的必要轨迹列"""

    cable = replace(recommended_cable(params), H=float(row["H"]), L=float(row["L"]))
    rows = []
    for mass in params.rider.mass_grid:
        sim = simulate_motion(
            cable,
            params.const,
            params.solver,
            mass=float(mass),
            xb_ratio=float(row["xb_ratio"]),
            FB=float(row["FB"]),
            resistance=params.resistance,
            v0=params.rider.v0,
            v_limit=params.rider.v_limit,
            a_safe=params.brake.a_safe,
        )
        jerk = np.gradient(sim.accel, sim.t, edge_order=1) if sim.t.size > 1 else np.zeros_like(sim.accel)
        for t, s, v, accel, j in zip(sim.t, sim.s, sim.v, sim.accel, jerk):
            rows.append(
                {
                    "mass": float(mass),
                    "t": float(t),
                    "s": float(s),
                    "v": float(v),
                    "accel": float(accel),
                    "jerk": float(j),
                }
            )
    return pd.DataFrame(rows)


def run(
    params: Params,
    population_size: int = 24,
    generations: int = 8,
    seed: int = 42,
    mutation_rate: float = 0.25,
) -> pd.DataFrame:
    """使用 NSGA-II 搜索综合优化参数"""
    rng = np.random.default_rng(seed)
    bounds = _bounds(params)
    population = _initial_population(params, population_size, rng)
    for i, ind in enumerate(population, start=1):
        print(f"[进度] NSGA-II 初始种群 {i}/{population_size}")
        _evaluate(ind, params, generation=0)
    population = _select_next(population, population_size)

    for generation in range(1, generations + 1):
        offspring: list[Individual] = []
        while len(offspring) < population_size:
            parent_a = _tournament(population, rng)
            parent_b = _tournament(population, rng)
            offspring.append(_make_child(parent_a, parent_b, bounds, rng, mutation_rate))
        for i, ind in enumerate(offspring, start=1):
            print(f"[进度] NSGA-II 第 {generation}/{generations} 代，子代 {i}/{population_size}")
            _evaluate(ind, params, generation=generation)
        population = _select_next(population + offspring, population_size)

    df = _rows(population)
    if not df.empty:
        df.attrs["plot_population"] = _population_rows(population)
        df.attrs["extra_tables"] = {
            "exp09_nsga2_best_trajectory.csv": _best_trajectory(params, df.iloc[0]),
        }
    return df


if __name__ == "__main__":
    run_experiment_outputs(
        "NSGA-II 综合优化",
        "exp09_nsga2_optimization.csv",
        lambda: run(Params(), population_size=12, generations=3),
        (plot_nsga2_pareto_front,),
    )
