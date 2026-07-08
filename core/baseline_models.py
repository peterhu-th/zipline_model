from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, root
from config.params import CableParams, PhysicalConstants


@dataclass
class AnalyticCatenary:
    # y(x) = a cosh((x-c) / a) + d
    a: float
    c: float
    d: float
    success: bool

    def y(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.a * np.cosh((np.asarray(x) - self.c) / self.a) + self.d

    def dydx(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.sinh((np.asarray(x) - self.c) / self.a)


@dataclass
class BaselineShapeResult:
    """基准索形模型的统一输出"""
    x: np.ndarray
    y: np.ndarray
    tension: np.ndarray
    T_max: float
    sag_max: float
    arc_length: float
    success: bool
    message: str = ""


@dataclass
class SmallExtensionResult:
    """小伸长修正悬链线结果"""
    model: AnalyticCatenary
    effective_length: float
    elastic_extension: float
    T_max: float
    sag_max: float
    success: bool
    iterations: int
    message: str = ""


def solve_analytic_catenary(cable: CableParams) -> AnalyticCatenary:
    """brentq 求根算法求参数组合"""
    D = float(np.hypot(cable.W, cable.H))
    if cable.L <= D:
        a = 1e9
        c = cable.W / 2.0
        d = cable.H / 2.0 - a
        return AnalyticCatenary(a=a, c=c, d=d, success=False)

    target = np.sqrt(max(cable.L * cable.L - cable.H * cable.H, 0.0)) / cable.W

    def f(m: float) -> float:
        return np.sinh(m) / m - target

    upper = 1.0
    while f(upper) < 0.0 and upper < 100.0:
        upper *= 2.0
    try:
        m = brentq(f, 1e-10, upper, maxiter=200)
        a = cable.W / (2.0 * m)
        n = -np.arcsinh(cable.H / (2.0 * a * np.sinh(m)))
        p = m + n
        q = m - n
        c = a * q
        d = -a * np.cosh(p)
        return AnalyticCatenary(a=float(a), c=float(c), d=float(d), success=True)
    except (ValueError, FloatingPointError, OverflowError):
        a = 1e9
        c = cable.W / 2.0
        d = cable.H / 2.0 - a
        return AnalyticCatenary(a=a, c=c, d=d, success=False)


def analytic_tension(model: AnalyticCatenary, cable: CableParams, const: PhysicalConstants) -> tuple[float, float]:
    """计算张力分布"""
    horizontal = cable.R * const.g * model.a
    xs = np.linspace(0.0, cable.W, 400)
    total = horizontal * np.sqrt(1.0 + np.asarray(model.dydx(xs)) ** 2)
    return float(horizontal), float(np.max(total))


def _catenary_sag(model: AnalyticCatenary, cable: CableParams, num: int = 400) -> float:
    """计算解析悬链线相对端点弦线的最大垂度"""
    xs = np.linspace(0.0, cable.W, num)
    chord = cable.H * (1.0 - xs / cable.W)
    return float(np.max(chord - model.y(xs)))


def _catenary_tension_samples(model: AnalyticCatenary, cable: CableParams, const: PhysicalConstants, num: int = 400) -> np.ndarray:
    """按解析悬链线采样张力"""
    xs = np.linspace(0.0, cable.W, num)
    horizontal = cable.R * const.g * model.a
    return horizontal * np.sqrt(1.0 + np.asarray(model.dydx(xs)) ** 2)


def solve_small_extension_model(
    cable: CableParams,
    const: PhysicalConstants | None = None,
    max_iter: int = 30,
    tol: float = 1e-7,
) -> SmallExtensionResult:
    """小伸长修正模型：迭代估计弹性伸长后的平衡弧长"""
    const = const or PhysicalConstants()
    effective_length = float(cable.L)
    last_extension = 0.0
    model = solve_analytic_catenary(cable)
    success = model.success
    message = ""

    for iteration in range(1, max_iter + 1):
        trial_cable = CableParams(
            W=cable.W,
            H=cable.H,
            L=effective_length,
            R=cable.R,
            E=cable.E,
            A_c=cable.A_c,
            N=cable.N,
            fixed_x=cable.fixed_x,
            T_allow=cable.T_allow,
            delta_allow=cable.delta_allow,
        )
        model = solve_analytic_catenary(trial_cable)
        if not model.success:
            success = False
            message = "解析悬链线求解失败"
            break

        tension = _catenary_tension_samples(model, trial_cable, const)
        extension = float(np.mean(tension) / max(cable.EA, 1e-12) * cable.L)
        next_length = cable.L + extension
        if abs(next_length - effective_length) <= tol * max(1.0, effective_length):
            effective_length = next_length
            success = True
            message = "小伸长修正收敛"
            break
        effective_length = next_length
        last_extension = extension
    else:
        iteration = max_iter
        success = False
        message = "小伸长修正达到最大迭代次数"

    final_cable = CableParams(
        W=cable.W,
        H=cable.H,
        L=effective_length,
        R=cable.R,
        E=cable.E,
        A_c=cable.A_c,
        N=cable.N,
        fixed_x=cable.fixed_x,
        T_allow=cable.T_allow,
        delta_allow=cable.delta_allow,
    )
    model = solve_analytic_catenary(final_cable)
    _, T_max = analytic_tension(model, final_cable, const)
    elastic_extension = max(0.0, effective_length - cable.L)
    if last_extension and not elastic_extension:
        elastic_extension = last_extension
    return SmallExtensionResult(
        model=model,
        effective_length=float(effective_length),
        elastic_extension=float(elastic_extension),
        T_max=float(T_max),
        sag_max=_catenary_sag(model, final_cable),
        success=bool(success and model.success),
        iterations=int(iteration),
        message=message,
    )


def _segment_y(x: np.ndarray, a: float, c: float, d: float) -> np.ndarray:
    return a * np.cosh((x - c) / a) + d


def _segment_slope(x: np.ndarray, a: float, c: float) -> np.ndarray:
    return np.sinh((x - c) / a)


def solve_segmented_catenary_model(
    cable: CableParams,
    rider_mass: float,
    rider_x: float,
    const: PhysicalConstants | None = None,
    num: int = 400,
) -> BaselineShapeResult:
    """载人静止状态的分段悬链线对照模型"""
    const = const or PhysicalConstants()
    xi = float(np.clip(rider_x, 1e-6 * cable.W, cable.W * (1.0 - 1e-6)))

    def equations(q: np.ndarray) -> np.ndarray:
        a, c1, d1, c2, d2, yp = q
        if a <= 0.0:
            return np.full(6, 1e6)
        y_left = a * np.cosh((xi - c1) / a) + d1
        y_right = a * np.cosh((xi - c2) / a) + d2
        s_left = a * (np.sinh((xi - c1) / a) - np.sinh((0.0 - c1) / a))
        s_right = a * (np.sinh((cable.W - c2) / a) - np.sinh((xi - c2) / a))
        jump = cable.R * const.g * a * (np.sinh((xi - c2) / a) - np.sinh((xi - c1) / a)) + rider_mass * const.g
        return np.array(
            [
                a * np.cosh((0.0 - c1) / a) + d1 - cable.H,
                a * np.cosh((cable.W - c2) / a) + d2,
                y_left - yp,
                y_right - yp,
                jump / max(cable.R * const.g * max(a, 1e-12), 1e-12),
                s_left + s_right - cable.L,
            ]
        )

    empty = solve_analytic_catenary(cable)
    a0 = empty.a if empty.success and np.isfinite(empty.a) else max(cable.W, 1.0)
    yp0 = cable.H * (1.0 - xi / cable.W) - 0.04 * cable.W
    q0 = np.array([a0, xi / 2.0, yp0 - a0, (xi + cable.W) / 2.0, yp0 - a0, yp0])
    result = root(equations, q0, method="hybr")

    if not result.success or result.x[0] <= 0.0:
        xs = np.linspace(0.0, cable.W, num)
        ys = empty.y(xs) if empty.success else cable.H * (1.0 - xs / cable.W)
        tension = np.zeros(max(num - 1, 1))
        return BaselineShapeResult(
            x=xs,
            y=np.asarray(ys),
            tension=tension,
            T_max=0.0,
            sag_max=0.0,
            arc_length=float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))),
            success=False,
            message=str(result.message),
        )

    a, c1, d1, c2, d2, _yp = result.x
    n_left = max(3, int(num * xi / cable.W))
    n_right = max(3, num - n_left + 1)
    xs_left = np.linspace(0.0, xi, n_left, endpoint=False)
    xs_right = np.linspace(xi, cable.W, n_right)
    xs = np.concatenate([xs_left, xs_right])
    ys = np.concatenate([_segment_y(xs_left, a, c1, d1), _segment_y(xs_right, a, c2, d2)])
    slopes = np.concatenate([_segment_slope(xs_left, a, c1), _segment_slope(xs_right, a, c2)])
    tension_nodes = cable.R * const.g * a * np.sqrt(1.0 + slopes * slopes)
    tension = (tension_nodes[:-1] + tension_nodes[1:]) / 2.0
    chord = cable.H * (1.0 - xs / cable.W)
    return BaselineShapeResult(
        x=xs,
        y=ys,
        tension=tension,
        T_max=float(np.max(tension_nodes)),
        sag_max=float(np.max(chord - ys)),
        arc_length=float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))),
        success=True,
        message="分段悬链线求解成功",
    )
