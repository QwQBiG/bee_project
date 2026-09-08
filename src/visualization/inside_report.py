"""生成无需额外前端依赖的巢内分析 HTML 报告。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict


GUIDANCE = {
    "个体朝向与特定运动轨迹": ("侦查蜂的舞蹈通讯", "验证外界植物开花情況，提前准备采蜜工具，并据此调整蜂箱朝向。"),
    "个体异常姿态": ("调节巢温行为（扇风排热）", "检查蜂箱遮阳、通风口和水源；候选异常仅作人工复核，不替代现场观察。"),
    "群体运动速度与活跃度": ("蜂群活力与健康度", "非冬季低活跃时检查蜜糖储备并酌情补饲；冬季则优先确认保温措施，避免频繁开箱。"),
    "局部空间异常高密度聚集": ("造王台或分蜂前兆", "检查是否存在王台；如需分蜂管理，应由养蜂人员结合现场开箱检查决定。"),
    "个体静止时间与掉落轨迹": ("病害与死亡风险", "检查是否有掉落或死亡个体；必要时排查螨害、幼虫腐臭病等并清理箱底。"),
}


def _value_lines(metric: Dict) -> str:
    skipped = {"name", "status", "description", "limitations", "grid"}
    lines = []
    for key, value in metric.items():
        if key in skipped:
            continue
        label = key.replace("_", " ")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"<li><span>{html.escape(label)}</span>{html.escape(str(value))}</li>")
    return "".join(lines) or "<li><span>结果</span>暂无</li>"


def create_inside_report(report: Dict, output_path: str | Path) -> Path:
    """写入中文 HTML 仪表板，浏览器直接打开即可查看。"""
    cards = []
    for metric in report.get("metrics", []):
        behavior, action = GUIDANCE.get(metric["name"], ("待进一步解释", "请结合现场观察复核。"))
        status = metric.get("status", "unknown")
        cards.append(f"""
        <article class=\"card {html.escape(status)}\">
          <div class=\"status\">{'需复核' if status == 'warning' else '正常' if status == 'normal' else '数据不足'}</div>
          <h2>{html.escape(metric['name'])}</h2><p>{html.escape(metric.get('description', ''))}</p>
          <ul>{_value_lines(metric)}</ul>
          <section><h3>推断出的蜜蜂行为（生物学意义）</h3><p>{html.escape(behavior)}</p>
          <h3>养蜂处理建议</h3><p>{html.escape(action)}</p></section>
          <small>{html.escape(metric.get('limitations', ''))}</small>
        </article>""")
    alert_text = "无自动预警" if not report.get("alerts") else "；".join(
        f"{item['metric']}（需复核）" for item in report["alerts"])
    document = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
    <title>巢内红外视频分析报告</title><style>
    *{{box-sizing:border-box}} body{{margin:0;background:#f5f7f4;color:#17201b;font:16px/1.65 -apple-system,BlinkMacSystemFont,'Microsoft YaHei',sans-serif}}
    main{{max-width:1440px;margin:auto;padding:42px 30px 56px}} header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:2px solid #d9e1d8;padding-bottom:22px}} h1{{font-size:31px;margin:0 0 5px}} h2{{font-size:21px;margin:10px 0}} h3{{font-size:15px;margin:18px 0 2px;color:#38503e}} p{{margin:4px 0}} .summary{{color:#607066}} .alert{{padding:10px 14px;border-radius:10px;background:#fff2e8;color:#9a3f00;font-weight:600}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;margin-top:26px}} .card{{position:relative;background:white;border:1px solid #e1e7e0;border-radius:14px;padding:25px;box-shadow:0 3px 14px #17301a0a}} .card.warning{{border-top:5px solid #db7a33}} .card.normal{{border-top:5px solid #4c9570}} .status{{position:absolute;right:20px;top:18px;font-size:13px;color:#5e7162}} .warning .status{{color:#b45112}} ul{{padding:0;margin:16px 0;list-style:none;border-top:1px solid #edf0ec}} li{{display:flex;justify-content:space-between;gap:15px;padding:7px 0;border-bottom:1px solid #edf0ec;word-break:break-all}} li span{{color:#68766b}} section{{background:#f7faf6;border-radius:9px;padding:10px 14px;margin-top:16px}} small{{display:block;color:#7c877e;margin-top:14px}} @media(max-width:650px){{main{{padding:24px 16px}}header{{display:block}}.alert{{margin-top:14px}}}}
    </style><main><header><div><h1>巢内场景（基于红外视频）</h1><div class=\"summary\">已分析 {report.get('frames_analyzed', 0)} 帧 · 产生 {report.get('tracked_individuals', 0)} 条轨迹</div></div><div class=\"alert\">{html.escape(alert_text)}</div></header><div class=\"grid\">{''.join(cards)}</div></main></html>"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    return target
