"""生物学指标的证据需求与可用性审查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, FrozenSet, Iterable, List


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    evidence_level: str
    required_signals: FrozenSet[str]
    supported_claim: str
    limitation: str


def _spec(key: str, name: str, evidence_level: str, signals: Iterable[str],
          claim: str, limitation: str) -> MetricSpec:
    return MetricSpec(key, name, evidence_level, frozenset(signals), claim, limitation)


METRIC_SPECS = (
    _spec("entrance_traffic", "进出巢数量", "direct_video",
          ["confirmed_entrance_events"], "统计给定时间窗内确认的进巢和出巢事件。",
          "未确认和缓冲区消失事件单独统计，不能强行归类。"),
    _spec("net_flow", "进出净差值", "direct_video",
          ["confirmed_entrance_events"], "计算确认进巢数减去确认出巢数。",
          "净差值不等同于蜂群死亡率或失蜂率。"),
    _spec("trajectory_speed", "轨迹与移动速度", "direct_video",
          ["calibrated_tracks", "video_fps"], "计算画面内轨迹、速度和停留时间。",
          "未做尺度标定时速度单位只能是像素/秒。"),
    _spec("activity_density", "群体活跃度与空间密度", "direct_video",
          ["calibrated_tracks"], "统计轨迹数量、移动强度和空间分布。",
          "检测漏失、遮挡和视野变化会影响结果。"),
    _spec("body_orientation", "头尾与身体朝向", "model_inference",
          ["validated_pose_model"], "输出头胸腹关键点、身体轴和朝向角。",
          "关键点缺失或遮挡时必须输出未知。"),
    _spec("pollen_load", "携粉进巢比例", "model_inference",
          ["validated_pollen_model", "confirmed_entrance_events"],
          "统计经人工验证模型识别的携粉进巢事件比例。",
          "颜色候选或 pollenbee 类别未评测前不能作为营养结论。"),
    _spec("orientation_flight", "认巢试飞候选", "model_inference",
          ["validated_pose_model", "calibrated_tracks", "expert_behavior_labels"],
          "筛选巢门前具有特定朝向和轨迹的候选片段。",
          "候选结果需要养蜂专家复核。"),
    _spec("dance_behavior", "舞蹈行为候选", "model_inference",
          ["validated_temporal_model", "validated_pose_model", "expert_behavior_labels"],
          "识别摇摆舞或圆圈舞候选时间片段。",
          "不能仅凭轨迹形状推断蜜源方向和距离。"),
    _spec("defense_behavior", "防卫行为候选", "model_inference",
          ["validated_temporal_model", "expert_behavior_labels"],
          "识别经定义和标注的防卫或冲突候选片段。",
          "无法仅凭高速度或聚集确认为盗蜂或天敌入侵。"),
    _spec("abnormal_posture", "异常姿态候选", "model_inference",
          ["validated_pose_model", "expert_behavior_labels"],
          "筛选相对正常姿态分布显著偏离的片段。",
          "异常姿态不等同于病害确诊。"),
    _spec("round_trip_time", "单一个体往返时间", "external_validation",
          ["persistent_individual_identity", "synchronized_timestamps"],
          "关联同一个体离巢和回巢时间。",
          "普通短时 track_id 不能支持跨画面或跨摄像头身份。"),
    _spec("survival_rate", "个体存活率", "external_validation",
          ["persistent_individual_identity", "longitudinal_observation", "mortality_ground_truth"],
          "在长期个体身份和死亡真值基础上估计存活率。",
          "静止或掉落轨迹不能直接判定死亡。"),
    _spec("source_distance", "蜜源距离与采集效率", "external_validation",
          ["persistent_individual_identity", "forage_source_ground_truth", "synchronized_timestamps"],
          "结合已知蜜源位置和个体往返记录分析采集效率。",
          "只有入口视频时无法得到真实蜜源距离。"),
    _spec("thermoregulation", "巢内温度调节", "external_validation",
          ["temperature_sensor", "synchronized_timestamps", "calibrated_tracks"],
          "联合温度序列和蜂群行为分析调温响应。",
          "普通近红外视频不是经过校准的温度测量。"),
    _spec("disease_risk", "病害风险候选", "external_validation",
          ["expert_disease_labels", "validated_disease_model"],
          "输出针对已标注病害表型的风险候选。",
          "风险候选不能替代养蜂专家或实验室诊断。"),
)


def assess_metric_readiness(available_signals: Iterable[str]) -> List[Dict]:
    available = set(available_signals)
    results = []
    for spec in METRIC_SPECS:
        missing = sorted(spec.required_signals - available)
        item = asdict(spec)
        item["required_signals"] = sorted(spec.required_signals)
        item["ready"] = not missing
        item["missing_signals"] = missing
        results.append(item)
    return results


def readiness_summary(available_signals: Iterable[str]) -> Dict:
    metrics = assess_metric_readiness(available_signals)
    ready = [item["key"] for item in metrics if item["ready"]]
    return {"ready_count": len(ready), "total_count": len(metrics),
            "ready_metrics": ready, "metrics": metrics}
