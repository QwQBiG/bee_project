"""占位数据模块。

巢外"轨迹质量模块"尚未开发，track_quality 返回占位数据，
implemented=False 供前端显示"开发中"标记。

其余分析结果均由真实推理产生（见 tasks.py → adapters.py）。
"""


def _outside_track_quality_placeholder() -> dict:
    """巢外轨迹连续性 / 稳定性指标（写死占位，未开发）。"""
    return {
        "implemented": False,
        "note": "轨迹质量模块尚未开发，以下为演示数据",
        "continuity": {
            "mean_track_length": 312.4,        # 平均轨迹长度（帧）
            "mean_track_lifetime_s": 10.4,     # 平均轨迹存活时长（秒）
            "track_break_rate": 0.06,          # 轨迹断裂率（max_age 淘汰占比）
        },
        "stability": {
            "mean_speed_std": 1.32,            # 平均速度标准差（px/frame）
            "direction_change_rate": 0.18,     # 方向变化率
            "missing_frame_ratio": 0.04,       # 轨迹点缺失率
        },
    }
