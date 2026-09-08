"""生成巢外采粉与营养趋势预测的单文件 HTML 报告。"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict


def _percentage(value) -> str:
    return "暂无" if value is None else f"{float(value):.1%}"


def create_outside_pollen_report(report: Dict, output_path: str | Path) -> Path:
    assessment = report.get("nutrition_assessment", {})
    status = assessment.get("status", "unknown")
    label = {
        "normal": "暂未发现明显风险",
        "warning": "建议重点复核",
        "unknown": "数据不足",
    }.get(status, "数据不足")
    confidence = {
        "high": "较高",
        "low": "较低",
        "insufficient": "不足",
    }.get(assessment.get("confidence"), "不足")
    inbound = report.get("inbound_events", 0)
    pollen_inbound = report.get("pollen_inbound_events", 0)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>巢外采粉与营养趋势预测</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f7f7f3;color:#182019;font:16px/1.65 'Microsoft YaHei',sans-serif}}
main{{max-width:1200px;margin:auto;padding:42px 26px}} h1{{margin:0}} .sub{{color:#637067}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin:25px 0}}
.card{{background:#fff;border-radius:13px;padding:22px;border:1px solid #e1e7df}}
.num{{font-size:32px;font-weight:700;color:#31583c}}
.warning{{border-top:5px solid #d9752c}} .normal{{border-top:5px solid #4f936c}}
.unknown{{border-top:5px solid #8b938d}} small{{color:#66716a}}
</style></head><body><main>
<h1>巢外场景：花粉采集与营养趋势预测</h1>
<p class="sub">方法：{html.escape(str(report.get('method', '')))}</p>
<div class="grid">
  <div class="card"><small>可分析轨迹</small><div class="num">{report.get('analyzable_tracks', 0)}</div></div>
  <div class="card"><small>携粉候选轨迹比例</small><div class="num">{_percentage(report.get('pollen_candidate_ratio'))}</div></div>
  <div class="card"><small>进巢事件 / 携粉进巢</small><div class="num">{inbound} / {pollen_inbound}</div></div>
  <div class="card"><small>携粉进巢比例</small><div class="num">{_percentage(report.get('pollen_inbound_ratio'))}</div></div>
</div>
<article class="card {html.escape(status)}">
  <h2>营养趋势预测：{label}</h2>
  <p><b>预测置信度：</b>{confidence}</p>
  <p>{html.escape(str(assessment.get('message', '')))}</p>
  <p><b>处理建议：</b>若连续多个可比时间窗均偏低，请核查外界花源与天气；确认花源不足后，再由养蜂人员决定是否补饲花粉饼。</p>
  <small>{html.escape(str(report.get('limitations', '')))}</small>
</article></main></body></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target
