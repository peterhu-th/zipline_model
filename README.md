# 钢缆溜索数学建模项目

本项目用于回答 `question.md` 中的钢缆溜索建模题。代码实现基于 `method.md` 的修订方法，按原题三问组织实验：

1. 空置与载人静止状态下的钢缆索形、垂度、张力分布，以及高差影响。
2. 人员沿钢缆滑行的动力学过程、最大速度、终点限速和缓冲参数设计。
3. 在给定景区条件下，对高差、线缆松紧、缓冲开始位置和缓冲摩擦力做鲁棒优化。

## 目录结构

```text
config/
  params.py                 参数、扫描网格和输出路径
core/
  baseline_models.py        解析悬链线等基准模型
  energy_cable.py           离散弹性索能量最小化静力模型
  geometry.py               插值、坡度、曲率、垂度和弧长计算
  dynamics.py               准静态移动载荷滑行动力学
  braking.py                缓冲参数可行性判定与扫描
  sensitivity.py            高差敏感性分析
experiments/
  exp01_convergence.py      网格收敛性实验
  exp02_static_model_compare.py 解析悬链线与能量模型对比
  exp03_height_tension.py   高差对张力和垂度的影响
  exp04_rider_position.py   载人位置对索形和张力的影响
  exp05_dynamic_model_compare.py 动力学烟测
  exp06_resistance_sensitivity.py 阻力参数敏感性
  exp07_braking_feasible_region.py 缓冲可行域扫描
  exp08_global_optimization.py 第三问综合参数优化
utils/
  io.py                     输出目录与 CSV 写入
  plotting.py               图像保存辅助
  validation.py             参数合法性检查
main.py                     统一运行入口
outputs/tables/             实验输出表格
```

## 主要方法

### 问题一：静力索形

空置钢缆以解析悬链线作为基准：

```text
y(x) = a cosh((x-c)/a) + d
```

主模型采用离散弹性索能量最小化。钢缆按未伸长原长 `L` 划分为若干单元，钢缆自重按未伸长线密度 `R` 计算，弹性伸长由轴向刚度 `EA_c` 决定。张力采用只受拉模型：

```text
T_i = EA_c max((l_i-l_{0i})/l_{0i}, 0)
```

默认 `fixed_x=True`，即水平节点等距、只优化节点高度，适合批量扫描。若需要精度验证，可在 `config/params.py` 中设置 `fixed_x=False`，同时优化中间节点的水平和竖直坐标。

### 问题二：滑行动力学

动力学采用准静态移动载荷：人员滑到某一位置时，先求该位置载人静力索形，再由局部坡度、曲率和阻力计算切向加速度。阻力包括滑动摩擦、空气阻力、等效线性阻尼和缓冲区恒定摩擦力。

终点安全约束为：

```text
v_T <= 3.5 m/s
```

若速度在到达终点前接近 0，则判定为中途滞留。

### 问题三：景区综合优化

设计变量包括：

```text
H, eta, x_b/W, F_B
```

其中：

```text
eta = (L - sqrt(W^2 + H^2)) / W
```

对 `30-80 kg` 体重范围做鲁棒约束，在满足终点速度、可达性、最大减速度、张力和垂度约束的基础上，对平均速度、峰值速度等体验指标排序。

## 环境依赖

```text
numpy
scipy
pandas
matplotlib
```

## 复现操作

运行快速实验，生成问题一和问题二的基础结果表：

```powershell
python main.py quick
```

运行第三问综合优化烟测：

```powershell
python main.py global --limit-per-axis 1
```

如果需要扩大搜索范围，可提高 `--limit-per-axis` 或直接运行完整网格：

```powershell
C:\ProgramData\miniconda3\envs\cable\python.exe main.py global
```
