"""命令行查看量规距离场与训练记录。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capture import load_capture
from .model import Box
from .training import TrainingLedger


def main(argv=None):
    parser = argparse.ArgumentParser(description="查看截断带符号距离场与训练记录")
    parser.add_argument("path", type=Path, help="采集交换记录 JSON")
    parser.add_argument("--box", nargs=4, type=int, metavar=("R0", "R1", "C0", "C1"),
                        help="完整幅面上的左闭右开窗口")
    parser.add_argument("--prior", action="store_true",
                        help="用上一个瓦片版本而不是最新版本")
    parser.add_argument("--gauge-change", action="store_true",
                        help="使用记录里的新量规")
    parser.add_argument("--probes", action="store_true", help="只输出定位点记录")
    parser.add_argument("--reissue-prior", action="store_true",
                        help="先按最新签发，再切换到上一版本并给出重发报告")
    settings = parser.parse_args(argv)

    capture = load_capture(settings.path)
    selection = capture.prior_selection() if settings.prior else None
    calibration = capture.gauge_change if settings.gauge_change else None
    field = capture.field(selection=selection, calibration=calibration)

    if settings.probes:
        ledger = TrainingLedger()
        ledger.issue(field, capture.probes)
        print(json.dumps([ledger.current(item).as_dict() for item in capture.probes],
                         ensure_ascii=False, indent=2))
        return 0

    if settings.reissue_prior:
        ledger = TrainingLedger()
        ledger.issue(field, capture.probes)
        prior = capture.field(selection=capture.prior_selection())
        report = ledger.reissue(field, prior, capture.probes)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    box = Box(*settings.box) if settings.box else capture.frame.box
    summary = field.summarize(box)
    summary["credentials"] = [list(item) for item in field.credentials]
    summary["calibration"] = field.calibration.version
    summary["spacing"] = list(field.gauge.spacing)
    summary["radius"] = field.gauge.radius
    summary["tile_shape"] = list(field.tile_shape)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
