"""生成巢外采粉与营养评估的单文件 HTML 报告。"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict


def create_outside_pollen_report(report: Dict, output_path: str | Path) -> Path:
    assessment = report.get("nutrition_assessment", {})
    status = assessment.get("status", "unknown")
    label = {"normal": "正常", "warning": "需复核", "unknown": "数据不足"}.get(status, "数据不足")
    ratio = report.get("pollen_inbound_ratio")
    ratio_text = "暂无" if ratio is None else f"{ratio:.1%}"
    document = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>巢外采粉分析报告</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#f7f7f3;color:#182019;font:16px/1.65 'Microsoft YaHei',sans-serif}}main{{max-width:1050px;margin:auto;padding:42px 26px}}h1{{margin:0}}.sub{{color:#637067}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin:25px 0}}.card{{background:#fff;border-radius:13px;padding:22px;border:1px solid #e1e7df}}.num{{font-size:32px;font-weight:700;color:#31583c}}.warning{{border-top:5px solid #d9752c}}.normal{{border-top:5px solid #4f936c}}.unknown{{border-top:5px solid #8b938d}}small{{color:#66716a}}</style><main><h1>巢外场景：花粉采集与营养评估</h1><p class=\"sub\">方法：{html.escape(str(report.get('method', '')))}</p><div class=\"grid\"><div class=\"card\"><small>可分析轨迹</small><div class=\"num\">{report.get('analyzable_tracks', 0)}</div></div><div class=\"card\"><small>进巢事件 / 携粉进巢</small><div class=\"num\">{report.get('inbound_events', 0)} / {report.get('pollen_inbound_events', 0)}</div></div><div class=\"card\"><small>携粉进巢比例</small><div class=\"num\">{ratio_text}</div></div></div><article class=\"card {html.escape(status)}\"><h2>营养评估：{label}</h2><p>{html.escape(assessment.get('message', ''))}</p><p><b>建议：</b>若连续多个可比时间窗均偏低，请核查外界花源与天气；确认花源不足后，再由养蜂人员决定是否补饲花粉饼。</p><small>{html.escape(str(report.get('limitations', '')))}</small></article></main></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target
