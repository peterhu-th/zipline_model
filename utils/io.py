from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from config.params import OUTPUT_DIR


COLUMN_NAME_CN = {
    "N": "离散单元数",
    "success": "是否收敛",
    "arc_length": "弧长",
    "T_max": "最大张力",
    "sag_max": "最大垂度",
    "objective": "目标函数值",
    "arc_length_rel_error": "弧长相对误差",
    "T_max_rel_error": "最大张力相对误差",
    "sag_max_rel_error": "最大垂度相对误差",
    "within_1_percent": "是否满足1%误差",
    "recommended_N": "推荐微元数",
    "case": "对比工况",
    "comparison_group": "对比组",
    "scenario": "对比场景",
    "reference_model": "参考模型",
    "compare_model": "对比模型",
    "reference_success": "参考模型是否收敛",
    "compare_success": "对比模型是否收敛",
    "reference_T_max": "参考最大张力",
    "compare_T_max": "对比最大张力",
    "reference_sag_max": "参考最大垂度",
    "compare_sag_max": "对比最大垂度",
    "state": "状态",
    "model": "模型",
    "xi_ratio": "载人位置比例",
    "mass": "质量",
    "H": "高度差",
    "dT_dH": "张力高度导数",
    "dSag_dH": "垂度高度导数",
    "S_T": "张力相对敏感度",
    "S_sag": "垂度相对敏感度",
    "rider_x": "载人水平位置",
    "rider_y": "载人处高度",
    "y": "索形高度",
    "sag": "垂度",
    "tension": "张力",
    "reached_terminal": "是否到达终点",
    "v_max": "最大速度",
    "x_vmax": "最大速度位置",
    "v_terminal": "终点速度",
    "terminal_s": "终点下滑位移",
    "terminal_v": "终点速度",
    "terminal_a": "终点加速度",
    "terminal_speed_safe": "是否满足终点限速",
    "failure_reason": "失败原因",
    "travel_time": "滑行时间",
    "max_accel": "最大加速度",
    "max_decel": "最大减速度",
    "avg_speed": "平均速度",
    "feasible": "是否可行",
    "message": "求解信息",
    "xb_ratio": "缓冲起始位置比例",
    "FB": "缓冲摩擦力",
    "mu": "滑动摩擦系数",
    "kd": "空气阻力系数",
    "ce": "等效阻尼系数",
    "scan_stage": "扫描阶段",
    "eta": "余长比例",
    "L": "钢缆原长",
    "score_avg_speed": "平均速度评分",
    "score_speed_s_over_t": "沿索平均速度目标",
    "max_abs_jerk": "最大冲击度",
    "max_braking_jerk": "最大制动冲击度",
    "v_T_30": "30kg终点速度",
    "v_T_80": "80kg终点速度",
    "v_max_30": "30kg最大速度",
    "v_max_80": "80kg最大速度",
    "t": "时间",
    "x": "水平位置",
    "s": "下滑位移",
    "v": "速度",
    "accel": "加速度",
    "jerk": "冲击度",
    "feasible_count": "可行组合数",
    "reached_count": "可达组合数",
    "not_reached_count": "未到达组合数",
    "terminal_speed_over_limit_count": "终点超速组合数",
    "min_terminal_speed": "最低终点速度",
    "reason": "原因分析",
}


def chinese_column_name(name: str) -> str:
    """把内部字段名转换为导出表格使用的中文表头"""

    if name in COLUMN_NAME_CN:
        return COLUMN_NAME_CN[name]
    match = re.fullmatch(r"v_avg_(.+)", name)
    if match:
        return f"{match.group(1)}kg平均速度"
    return name


def localize_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    """生成中文表头副本，不改变实验内部 DataFrame"""

    return df.rename(columns={name: chinese_column_name(str(name)) for name in df.columns})


def ensure_output_dirs() -> None:
    for name in ("figures", "tables", "logs"):
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def write_table(df: pd.DataFrame, name: str) -> Path:
    ensure_output_dirs()
    path = OUTPUT_DIR / "tables" / name
    localize_table_columns(df).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.3e")
    return path
