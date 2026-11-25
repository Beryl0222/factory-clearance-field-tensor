"""从采集资料构造瓦片、标定与距离场。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .field import DEFAULT_SCRATCH_BYTES, SignedDistanceField
from .model import (
    Box,
    Calibration,
    Frame,
    Gauge,
    LabelMap,
    OwnerRegion,
    Tile,
    TileInventory,
    TileRecord,
)
from .training import FieldVersion, TrainingRecord, evaluate_probe, field_version

DEFAULT_TILE_SHAPE = (96, 64)


@dataclass
class Capture:
    """一份解析后的采集资料。"""

    frame: Frame
    calibration: Calibration
    inventory: TileInventory
    owners: list[OwnerRegion]
    probes: dict[str, tuple[float, float]]
    gauge_change: Calibration | None = None
    history: list[int] | None = None
    records: dict | None = None

    def field(self, *, calibration: Calibration | None = None,
              selection: dict[str, int] | None = None,
              tile_shape: tuple[int, int] = DEFAULT_TILE_SHAPE,
              scratch_bytes: int = DEFAULT_SCRATCH_BYTES) -> SignedDistanceField:
        labels = LabelMap(self.frame, self.owners, self.inventory, selection)
        return SignedDistanceField(
            labels,
            self.calibration if calibration is None else calibration,
            tile_shape=tile_shape,
            scratch_bytes=scratch_bytes,
        )

    def owner_of(self, row: int, col: int) -> OwnerRegion | None:
        for region in self.owners:
            if region.contains(row, col):
                return region
        return None

    def latest_selection(self) -> dict[str, int]:
        return self.inventory.selection()

    def prior_selection(self) -> dict[str, int]:
        """去掉每个 ``id`` 最新版本后的选择，用于按旧凭据回查。"""
        chosen = {}
        for tile_id, versions in self.inventory.tiles.items():
            ordered = sorted(versions)
            chosen[tile_id] = ordered[-2] if len(ordered) > 1 else ordered[-1]
        return chosen

    def tiles(self, *, latest: bool = True) -> list[Tile]:
        selection = self.latest_selection() if latest else self.prior_selection()
        return self.inventory.selected_tiles(selection)


def _arrays(envelope: dict) -> dict[str, np.ndarray]:
    arrays = {}
    for key, descriptor in envelope["tensors"].items():
        value = np.asarray(descriptor["values"], dtype=descriptor["dtype"])
        if tuple(descriptor["shape"]) != value.shape:
            raise ValueError(f"张量 {key} 与声明形状不一致")
        arrays[key] = value
    return arrays


def build_capture(envelope: dict) -> Capture:
    """从交换记录构造 :class:`Capture`。"""
    records = envelope["records"]
    arrays = _arrays(envelope)

    rows, cols = (int(v) for v in records["grid"])
    frame = Frame(rows, cols)

    spacing = tuple(int(v) for v in records["spacing"])
    calibration = Calibration(int(records.get("version", 1)),
                              Gauge(spacing, int(records["radius"])))

    gauge_change = None
    if records.get("gauge_change"):
        change = records["gauge_change"]
        gauge_change = Calibration.from_record(change["version"], change["spacing"], change["radius"])

    owners = [
        OwnerRegion.from_record(source, values)
        for source, values in records.get("owners", {}).items()
    ]

    inventory = TileInventory()
    for entry in records.get("tiles", []):
        record = TileRecord.from_record(entry)
        tensor = arrays[record.tensor]
        inventory.add(Tile.from_record(entry, tensor))

    probes = {
        str(item["id"]): (float(item["physical"][0]), float(item["physical"][1]))
        for item in records.get("probes", [])
    }

    return Capture(
        frame=frame,
        calibration=calibration,
        inventory=inventory,
        owners=owners,
        probes=probes,
        gauge_change=gauge_change,
        history=[int(v) for v in records.get("history", [])],
        records=records,
    )


def load_capture(path: Path) -> Capture:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_capture(envelope)
