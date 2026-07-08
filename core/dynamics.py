from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from config.params import CableParams, PhysicalConstants, ResistanceParams, SolverParams
from core.energy_cable import CableShape, solve_cable_shape
from core.geometry import curvature, interp_y, max_sag, slope


@dataclass
class SimulationResult:
    """一次滑行动力学仿真的汇总结果"""

    t: np.ndarray
    x: np.ndarray
    s: np.ndarray
    v: np.ndarray
    accel: np.ndarray
    reached_terminal: bool
    stalled: bool
    v_max: float
    x_vmax: float
    v_terminal: float
    travel_time: float
    max_accel: float
    max_decel: float
    T_max: float
    sag_max: float
    arc_length: float
    avg_speed: float
    feasible: bool
    message: str = ""


class ShapeCache:
    """准静态移动载荷缓存

    积分过程中，每到一个位置都要解一次静力索形。把位置按 cache_dx 分桶，并用上一次索形热启动
    """

    def __init__(self, cable: CableParams, const: PhysicalConstants, solver: SolverParams, mass: float):
        self.cable = cable
        self.const = const
        self.solver = solver
        self.mass = mass
        self.cache: dict[float, CableShape] = {}
        self.last_shape: CableShape | None = None

    def get(self, x: float) -> CableShape:
        key = round(float(np.clip(x, 0.0, self.cable.W)) / self.solver.cache_dx) * self.solver.cache_dx
        if key not in self.cache:
            self.last_shape = solve_cable_shape(
                self.cable,
                self.const,
                self.solver,
                rider_mass=self.mass,
                rider_x=key,
                initial=self.last_shape,
            )
            self.cache[key] = self.last_shape
        return self.cache[key]


def arc_displacement(reference_shape: CableShape, x: np.ndarray) -> np.ndarray:
    """沿参考索形计算弧长位移 s(x)"""

    if x.size == 0:
        return np.array([])
    values = np.zeros_like(x, dtype=float)
    for i in range(1, x.size):
        x0 = float(x[i - 1])
        x1 = float(x[i])
        y0 = slope(reference_shape, x0)
        y1 = slope(reference_shape, x1)
        ds_dx0 = np.sqrt(1.0 + y0 * y0)
        ds_dx1 = np.sqrt(1.0 + y1 * y1)
        values[i] = values[i - 1] + 0.5 * (ds_dx0 + ds_dx1) * max(0.0, x1 - x0)
    return values


def local_acceleration(
    x: float,
    v: float,
    shape: CableShape,
    cable: CableParams,
    const: PhysicalConstants,
    mass: float,
    mu: float,
    kd: float,
    ce: float,
    xb_ratio: float,
    FB: float,
) -> float:
    """计算给定位置和速度下的切向加速度

    驱动力为重力切向分量，阻力包括库仑摩擦、二次空气阻力、等效线性阻尼和缓冲区内的恒定摩擦力
    """

    yx = slope(shape, x)
    kappa = curvature(shape, x)
    denom = np.sqrt(1.0 + yx * yx)
    sin_alpha = -yx / denom
    cos_alpha = 1.0 / denom
    normal = max(0.0, mass * const.g * cos_alpha + mass * v * v * kappa)
    brake = FB if x >= xb_ratio * cable.W else 0.0
    force = mass * const.g * sin_alpha - mu * normal - kd * v * abs(v) - ce * v - brake
    return float(force / mass)


def simulate_motion(
    cable: CableParams,
    const: PhysicalConstants,
    solver: SolverParams,
    mass: float,
    xb_ratio: float,
    FB: float,
    resistance: ResistanceParams,
    v0: float = 0.05,
    v_limit: float = 3.5,
    a_safe: float = 3.0,
    mu: float | None = None,
    kd: float | None = None,
    ce: float | None = None,
) -> SimulationResult:
    """积分人员沿钢缆滑行的 x(t), v(t)。"""

    mu = resistance.mu_default if mu is None else mu
    kd = resistance.kd_default if kd is None else kd
    ce = resistance.ce_default if ce is None else ce
    cache = ShapeCache(cable, const, solver, mass)
    reference_shape = solve_cable_shape(cable, const, solver)

    return _simulate_with_shape_provider(
        cable,
        const,
        solver,
        mass,
        xb_ratio,
        FB,
        resistance,
        v0,
        v_limit,
        a_safe,
        mu,
        kd,
        ce,
        cache.get,
        reference_shape,
    )


def simulate_motion_fixed_shape(
    cable: CableParams,
    const: PhysicalConstants,
    solver: SolverParams,
    mass: float,
    xb_ratio: float,
    FB: float,
    resistance: ResistanceParams,
    shape: CableShape,
    v0: float = 0.05,
    v_limit: float = 3.5,
    a_safe: float = 3.0,
    mu: float | None = None,
    kd: float | None = None,
    ce: float | None = None,
) -> SimulationResult:
    """固定索形动力学，用于和准静态移动载荷模型对照"""

    mu = resistance.mu_default if mu is None else mu
    kd = resistance.kd_default if kd is None else kd
    ce = resistance.ce_default if ce is None else ce
    return _simulate_with_shape_provider(
        cable,
        const,
        solver,
        mass,
        xb_ratio,
        FB,
        resistance,
        v0,
        v_limit,
        a_safe,
        mu,
        kd,
        ce,
        lambda _x: shape,
        shape,
    )


def _simulate_with_shape_provider(
    cable: CableParams,
    const: PhysicalConstants,
    solver: SolverParams,
    mass: float,
    xb_ratio: float,
    FB: float,
    resistance: ResistanceParams,
    v0: float,
    v_limit: float,
    a_safe: float,
    mu: float,
    kd: float,
    ce: float,
    shape_at,
    reference_shape: CableShape,
) -> SimulationResult:
    """按给定索形提供器执行统一动力学积分"""

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        # 使用当前位置对应的准静态索形计算局部坡度和切向加速度。
        x = float(np.clip(z[0], 0.0, cable.W))
        v = max(float(z[1]), 0.0)
        shape = shape_at(x)
        yx = slope(shape, x)
        dxdt = v / np.sqrt(1.0 + yx * yx)
        dvdt = local_acceleration(x, v, shape, cable, const, mass, mu, kd, ce, xb_ratio, FB)
        return np.array([dxdt, dvdt])

    def terminal_event(_t: float, z: np.ndarray) -> float:
        return float(z[0] - cable.W)

    terminal_event.terminal = True
    terminal_event.direction = 1

    def stall_event(_t: float, z: np.ndarray) -> float:
        # 速度接近 0 且尚未到达终点时，判定为中途滞留。
        x = float(np.clip(z[0], 0.0, cable.W))
        if x < 1.0 or x >= cable.W - 1e-6:
            return 1.0
        v = max(float(z[1]), 0.0)
        shape = shape_at(x)
        accel = local_acceleration(x, v, shape, cable, const, mass, mu, kd, ce, xb_ratio, FB)
        if accel > 1e-5:
            return 1.0
        return float(z[1] - 1e-3)

    stall_event.terminal = True
    stall_event.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, 1000.0),
        np.array([0.0, v0]),
        rtol=solver.rtol,
        atol=solver.atol,
        max_step=solver.max_step,
        events=[terminal_event, stall_event],
    )
    x = np.clip(sol.y[0], 0.0, cable.W)
    s = arc_displacement(reference_shape, x)
    v = np.maximum(sol.y[1], 0.0)
    acc = np.empty_like(v)
    T_max = 0.0
    sag_max_value = 0.0
    for i, (xi, vi) in enumerate(zip(x, v)):
        shape = shape_at(float(xi))
        acc[i] = local_acceleration(float(xi), float(vi), shape, cable, const, mass, mu, kd, ce, xb_ratio, FB)
        T_max = max(T_max, shape.T_max)
        sag_max_value = max(sag_max_value, max_sag(shape, cable))

    reached = bool(sol.t_events[0].size)
    last_accel = float(acc[-1]) if acc.size else 0.0
    stalled = (bool(sol.t_events[1].size) and not reached) or (
        not reached and x[-1] < cable.W and v[-1] <= 1e-2 and last_accel <= 1e-5
    )
    imax = int(np.argmax(v)) if v.size else 0
    v_terminal = float(v[-1]) if reached else float("nan")
    max_decel = float(max(0.0, -np.min(acc))) if acc.size else 0.0
    max_accel = float(np.max(acc)) if acc.size else 0.0
    arc_length = float(s[-1]) if s.size else 0.0
    avg_speed = arc_length / float(sol.t[-1]) if sol.t.size and sol.t[-1] > 0.0 else float("nan")
    feasible = reached and not stalled and v_terminal <= v_limit and max_decel <= a_safe
    return SimulationResult(
        t=sol.t,
        x=x,
        s=s,
        v=v,
        accel=acc,
        reached_terminal=reached,
        stalled=stalled,
        v_max=float(np.max(v)) if v.size else 0.0,
        x_vmax=float(x[imax]) if x.size else 0.0,
        v_terminal=v_terminal,
        travel_time=float(sol.t[-1]) if sol.t.size else float("nan"),
        max_accel=max_accel,
        max_decel=max_decel,
        T_max=float(T_max),
        sag_max=float(sag_max_value),
        arc_length=arc_length,
        avg_speed=float(avg_speed),
        feasible=feasible,
        message=str(sol.message),
    )
