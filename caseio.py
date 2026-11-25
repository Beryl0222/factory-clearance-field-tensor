"""离线读取合成采集资料。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_case(path: Path):
    """读出交接记录和原始张量。"""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    arrays = {}
    for key, descriptor in envelope["tensors"].items():
        value = np.asarray(descriptor["values"], dtype=descriptor["dtype"])
        if tuple(descriptor["shape"]) != value.shape:
            raise ValueError(f"张量 {key} 与声明形状不一致")
        arrays[key] = value
    return envelope["records"], arrays


def main():
    parser = argparse.ArgumentParser(description="查看本地采集记录与张量尺寸")
    parser.add_argument("path", type=Path)
    parser.add_argument("--field", action="store_true",
                        help="额外打印最新瓦片下的距离面汇总")
    parser.add_argument("--prior", action="store_true",
                        help="用上一个瓦片版本而不是最新版本")
    parser.add_argument("--gauge-change", action="store_true",
                        help="使用记录里的新量规")
    settings = parser.parse_args()
    records, arrays = load_case(settings.path)
    print("记录字段", "、".join(sorted(records)))
    for key, array in sorted(arrays.items()):
        print(key, "形状", list(array.shape), "类型", array.dtype)
    if settings.field or settings.prior or settings.gauge_change:
        from gauge_distance.capture import build_capture

        capture = build_capture(json.loads(settings.path.read_text(encoding="utf-8")))
        selection = capture.prior_selection() if settings.prior else None
        calibration = capture.gauge_change if settings.gauge_change else None
        field = capture.field(selection=selection, calibration=calibration)
        print("瓦片凭据", field.credentials)
        print("量规", field.calibration.version, field.gauge.spacing,
              "半径", field.gauge.radius)
        print("距离面", json.dumps(field.summarize(capture.frame.box),
                                   ensure_ascii=False))


if __name__ == "__main__":
    main()
