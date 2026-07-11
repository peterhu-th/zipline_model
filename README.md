# 钢缆溜索数学建模项目

## 问题

很多景区设置了钢缆溜索项目，溜索由两端固定的主钢缆、滑动吊具组成。钢缆有自重和弹性，空置时呈自然悬链线形态，载人时会发生形变，产生挠度与水平落差。游客觉得滑动速度越快越有激情，不过为确保安全，人员到达末端时速度不能大于3.5米/秒。已知两端距离、高差、钢缆长度、钢缆线密度和弹性模量、滑动吊具的减速设计等，都会影响到人员在溜索上的滑行速度。

请建立数学模型回答如下问题：

1、假设已知两岸钢缆固定点之间的水平距离W、固定点高差H、钢缆总长L、钢缆线密度R、弹性模量E、人员体重w_0。不考虑空气阻力、风力影响，请给出空置、载人(人员停在溜索上某个位置)两种状态下的悬链线方程，求解钢缆任意位置垂度、张力分布，分析两岸高差对钢缆张力与垂度的影响。

2、考虑重力、钢缆弹性阻力、空气阻力和滑动摩擦，建立体重w_0的人员沿钢缆下滑的位移随时间变化的函数，计算人员在溜索上的最大滑行速度和达到最大滑行速度的位置。从安全的角度出发，人员应该在滑到目标位置时速度不大于3.5米/秒。为保证安全，滑动吊具应在什么位置开始，增加多大的恒定缓冲摩擦力才能保障人员安全？注意，减速太早可能导致人员停在中间、无法到达终点，减速太快可能导致人员在空中剧烈甩动出现危险。

3、假设某景区出发站与到达站的水平跨度W=600米，高差H在5-20米可调，钢缆线密度R=1.2千克/米，只接待体重30-80千克的游客。景区应该如何设置高差、线缆角度(拉紧或放松可调节)、缓冲开始位置、缓冲摩擦力大小，才能让游客既满意又安全？

## 目录结构

```text
config/
  params.py                         参数、扫描网格和输出路径
core/
  baseline_models.py                解析悬链线、小伸长修正、分段悬链线对照模型
  energy_cable.py                   离散弹性索能量最小化静力模型
  geometry.py                       插值、坡度、曲率、垂度和弧长计算
  dynamics.py                       准静态移动载荷滑行动力学
  braking.py                        缓冲参数可行性判断与扫描
  sensitivity.py                    高度差敏感性分析
experiments/
  common.py                         推荐微元数等共享实验逻辑
  exp01_convergence.py              微元数收敛性实验
  exp02_static_model_compare.py     静力模型误差对比
  exp03_height_tension.py           高度差对张力和垂度的影响
  exp04_rider_position.py           空置/载人任意位置垂度和张力分布
  exp05_dynamic_model_compare.py    动力学模型对比和运动轨迹
  exp06_resistance_sensitivity.py   阻力参数敏感性实验
  exp07_braking_feasible_region.py  缓冲可行域扫描
  exp08_global_optimization.py      网格遍历综合优化
  exp09_nsga2_optimization.py       NSGA-II 多目标综合优化
utils/
  experiment_runner.py              统一运行、写表、绘图入口
  io.py                             中文表头和 CSV 导出
  plotting.py                       可复用绘图函数
paper/
  main.tex                          论文 LaTeX 源文件
main.py                             统一实验入口
outputs/
  tables/                           实验表格输出
  figures/                          实验图像输出
```

## 环境

当前依赖记录在 `requirements.txt` 中：

```text
python=3.12.13
libffi=3.4.4
numpy=2.5.1
pandas=3.0.3
scipy=1.18.0
matplotlib=3.11.0
```

## 模型概述

### 静力模型

空置状态以解析悬链线作为基准：

```text
y(x) = a cosh((x-c)/a) + d
```

主模型将钢缆按未伸长原长划分为 `N` 个单元，节点高度由总势能最小化得到。总势能包括钢缆自重势能、弹性势能和人员集中载荷势能。张力按单元伸长计算：

```text
T_i = EA max((l_i-l_0i)/l_0i, 0)
```

载人对照模型采用分段悬链线，在人员位置满足位移连续、水平张力一致和竖向力跳跃条件。

### 动力学模型

人员沿钢缆滑行时，程序在当前位置求解载人准静态索形，再根据局部坡度和曲率计算切向加速度。受力包括：

- 重力切向分量
- 滑动摩擦
- 二次空气阻力
- 等效线性阻尼
- 缓冲区内恒定附加摩擦力

运动结果记录 `t,s,v,a`，其中 `s` 为沿索弧长位移，不是水平位移。

### 优化模型

综合优化变量为：

```text
H, eta, xb_ratio, FB
```

其中 `eta` 为余长比例，`xb_ratio` 为缓冲开始位置比例，`FB` 为恒定缓冲摩擦力。

项目提供两类优化：

- `exp08`：粗网格遍历 + 局部细化，结果容易解释。
- `exp09`：NSGA-II 多目标优化，适合在连续变量空间中寻找速度、舒适性和安全性的 Pareto 折中方案。

## 实验说明

| 实验  | 作用                                   | 主要输出                                                         |
| ----- | -------------------------------------- | ---------------------------------------------------------------- |
| exp01 | 寻找满足 1% 张力/垂度误差的微元数边界  | `exp01_convergence.csv`                                          |
| exp02 | 比较解析悬链线、分段悬链线和能量模型   | `exp02_static_model_compare.csv`                                 |
| exp03 | 分析高度差对最大张力和最大垂度的敏感性 | `exp03_height_tension.csv`、H-T/H-垂度图                         |
| exp04 | 输出空置和载人状态的任意位置垂度与张力 | `exp04_static_distribution.csv`、分布曲线                        |
| exp05 | 比较动力学模型并输出运动轨迹           | `exp05_dynamic_model_compare.csv`、`exp05_motion_trajectory.csv` |
| exp06 | 分析摩擦、空气阻力、阻尼参数影响       | `exp06_resistance_sensitivity.csv`                               |
| exp07 | 扫描缓冲起点和恒定缓冲力可行域         | `exp07_braking_feasible_region.csv`、热力图                      |
| exp08 | 网格遍历综合优化                       | `exp08_global_optimization.csv`、最优轨迹附表                    |
| exp09 | NSGA-II 多目标优化                     | `exp09_nsga2_optimization.csv`、最优轨迹附表、Pareto 图          |

## 运行方式

快速运行基础实验：

```powershell
python main.py quick
```

运行单个实验：

```powershell
python main.py convergence
python main.py rider_position
python main.py braking
```

运行网格综合优化烟测：

```powershell
python main.py global --limit-per-axis 1
```

运行 NSGA-II 优化：

```powershell
python main.py nsga2 --nsga-population 24 --nsga-generations 8 --seed 42
```

较大的种群和代数会显著增加计算时间。建议先用较小参数烟测，再扩大搜索规模。
