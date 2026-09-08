"""根据现有证据输出可声明的生物学指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from behavior.metric_registry import readiness_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="审查生物学指标所需证据是否齐全")
    parser.add_argument("signals", nargs="*", help="当前已经验证的信号名称")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = readiness_summary(args.signals)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
