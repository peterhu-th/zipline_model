from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class PhysicalConstants:
    """全局物理常数"""

    g: float = 9.8          # 重力加速度
    rho_air: float = 1.225  # 空气密度


@dataclass(frozen=True)
class CableParams:
    """钢缆静力模型参数。

    L 按未伸长原长解释；弹性伸长实际由 EA=E*A_c 决定。
    fixed_x=True 是快速计算模式，只优化节点高度；fixed_x=False 会同时优化
    中间节点的水平和竖直坐标，用于精度验证但计算更慢。
    """
    W: float = 600.0            # 跨度
    H: float = 15.0             # 高度差
    L: float = 604.0            # 钢缆长度
    R: float = 1.2              # 线密度
    E: float = 2.0e11           # 弹性模量
    A_c: float = 2.5e-4         # 截面积
    N: int = 50                 # 离散单元数
    fixed_x: bool = True        # 是否仅优化节点高度模式
    T_allow: float = 2.0e6      # 允许最大张力
    delta_allow: float = 80.0   # 允许最大

    @property
    def EA(self) -> float:
        return self.E * self.A_c


@dataclass(frozen=True)
class RiderParams:
    """游客质量与速度安全约束"""
    mass_grid: tuple[float, ...] = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0) # 游客质量
    v_limit: float = 3.5                                                # 终点限速
    v0: float = 0.05                                                    # 初始速度


@dataclass(frozen=True)
class ResistanceParams:
    """滑动摩擦、空气阻力和等效阻尼参数"""
    mu_default: float = 0.003                               # 滑动摩擦系数默认值
    kd_default: float = 0.01                                # 二次空气阻力系数默认值
    ce_default: float = 0.10                                # 线性阻尼系数默认值
    mu_grid: tuple[float, ...] = (0.0, 0.002, 0.003, 0.005, 0.01)
    kd_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
    ce_grid: tuple[float, ...] = (0.0, 0.10, 0.50, 1.0, 2.0)


@dataclass(frozen=True)
class BrakeParams:
    """缓冲设计变量和制动平顺性约束"""
    xb_ratio_grid: tuple[float, ...] = (0.75, 0.80, 0.85, 0.90, 0.92, 0.94)     # 缓冲开始位置比例
    FB_grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0, 501, 20)) # 缓冲区附加摩擦力
    a_safe: float = 3.0                                                         # 安全最大速度


@dataclass(frozen=True)
class SolverParams:
    """数值优化和 ODE 积分参数"""
    optimizer_method: str = "SLSQP" # 非线性优化算法
    max_iter: int = 400             # 最大迭代次数
    ftol: float = 1e-8              # 目标函数相对容差
    rtol: float = 1e-7              # ODE 积分相对容差
    atol: float = 1e-9              # ODE 积分绝对容差
    max_step: float = 1.0           # ODE 积分最大步长
    diff_H: float = 0.1             # 敏感性分析差分计算的微小扰动量
    cache_dx: float = 20.0          # 位移缓存分桶区间


@dataclass(frozen=True)
class ExperimentGrid:
    """实验用默认扫描网格"""
    H_grid: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0)                        # 垂直高度优化测试网格
    eta_grid: tuple[float, ...] = (0.002, 0.004, 0.006, 0.008, 0.010, 0.015)                    # 余长比例优化测试网格
    N_grid: tuple[int, ...] = (30, 50, 80)                                                      # 空间离散单元数收敛性测试网格
    rider_pos_ratio_grid: tuple[float, ...] = tuple(float(x) for x in np.arange(0.1, 1.0, 0.1)) # 载人位置占总跨距比例测试网格


@dataclass(frozen=True)
class Params:
    const: PhysicalConstants = field(default_factory=PhysicalConstants)
    cable: CableParams = field(default_factory=CableParams)
    rider: RiderParams = field(default_factory=RiderParams)
    resistance: ResistanceParams = field(default_factory=ResistanceParams)
    brake: BrakeParams = field(default_factory=BrakeParams)
    solver: SolverParams = field(default_factory=SolverParams)
    grid: ExperimentGrid = field(default_factory=ExperimentGrid)


def cable_length_from_eta(W: float, H: float, eta: float) -> float:
    """由余长比例 eta 计算未伸长索长"""
    return float(np.hypot(W, H) + eta * W)


DEFAULT_PARAMS = Params()
