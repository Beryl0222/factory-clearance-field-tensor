"""采集员更正、瓦片管辖与工作区规模。"""
import json
import random
from pathlib import Path
import tracemalloc
import unittest

import numpy as np

from gauge_distance.capture import load_capture
from gauge_distance.field import FieldStatus, SignedDistanceField, Uncertainty
from gauge_distance.model import (
    Box,
    Calibration,
    Frame,
    Gauge,
    LabelMap,
    OwnerRegion,
    Tile,
    TileInventory,
)
from gauge_distance.training import (
    TrainingError,
    TrainingLedger,
    evaluate_probe,
    record_probe,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CAPACITY = json.loads((DATA / "capacity.json").read_text(encoding="utf-8"))


def build(labels, spacing, radius, tile_shape=None, scratch=1 << 30):
    labels = np.asarray(labels, dtype=np.int8)
    rows, cols = labels.shape
    frame = Frame(rows, cols)
    tile = Tile("all", "s", 1, Box(0, rows, 0, cols),
                tuple(tuple(int(v) for v in line) for line in labels))
    inventory = TileInventory()
    inventory.add(tile)
    label_map = LabelMap(frame, [OwnerRegion("s", 0, rows, 0, cols)], inventory)
    return SignedDistanceField(
        label_map, Calibration(1, Gauge(spacing, radius)),
        tile_shape=(rows, cols) if tile_shape is None else tile_shape,
        scratch_bytes=scratch,
    )


class CorrectionTests(unittest.TestCase):
    def test_erasing_a_witness_needs_requalification(self):
        labels = np.array([
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=np.int8)
        before = build(labels, (1, 1), 10)
        record = record_probe(before, "p", 1, 1)
        self.assertEqual(record.witness.row, 0)
        self.assertEqual(record.witness.col, 2)

        corrected = labels.copy()
        corrected[0, 2] = -1  # 采集员擦掉那块数据
        after = build(corrected, (1, 1), 10)
        fresh = evaluate_probe(after, "p", 1, 1)
        self.assertEqual(fresh.status, FieldStatus.UNDETERMINED)
        self.assertEqual(fresh.uncertainty, Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS)
        self.assertFalse(fresh.eligible)
        with self.assertRaises(TrainingError):
            record_probe(after, "p", 1, 1)
        from gauge_distance.training import verify_record
        with self.assertRaises(TrainingError):
            verify_record(record, after)

    def test_patching_a_nearer_witness_is_a_valid_correction(self):
        labels = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.int8)
        before = build(labels, (1, 1), 10)
        record = record_probe(before, "p", 1, 1)
        self.assertEqual(record.squared, 2)

        corrected = labels.copy()
        corrected[1, 2] = 0  # 补入更近的背景
        after = build(corrected, (1, 1), 10)
        fresh = record_probe(after, "p", 1, 1)
        self.assertEqual(fresh.squared, 1)
        self.assertEqual((fresh.witness.row, fresh.witness.col), (1, 2))
        self.assertNotEqual(fresh.ticket, record.ticket)

    def test_ledger_links_correction_to_previous_ticket(self):
        labels = np.array([[1, 1, 0, 1]], dtype=np.int8)
        field = build(labels, (1, 1), 10)
        ledger = TrainingLedger()
        ledger.issue(field, {"p": (0, 1)})

        corrected = labels.copy()
        corrected[0, 1] = 0
        corrupted = build(corrected, (1, 1), 10)
        ledger.reissue(field, corrupted, {"p": (0, 1)})
        current = ledger.current("p")
        history = ledger.history_of("p")
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[0].superseded_by, current.ticket)
        explanation = ledger.explain("p")
        self.assertTrue(any(item["action"] in ("reissued", "lost")
                            for item in explanation["entries"]))

    def test_distances_unchanged_but_witness_invalidated_forces_new_version(self):
        # 等距两格之间的见证被擦除：距离数值不变，但下游凭据必须换版。
        labels = np.array([
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.int8)
        before = build(labels, (1, 1), 10)
        # 查询格 (0,0) 的等距背景见证有两个：(0,1) ID 1 与 (1,0) ID 3，取 ID 小的。
        record = record_probe(before, "p", 0, 0)
        self.assertEqual(record.squared, 1)
        self.assertEqual(record.witness.identity, 1)

        # 采集员把 (0,1) 补成前景：距离数值仍是 1，但见证换成 (1,0)。
        corrected = labels.copy()
        corrected[0, 1] = 1
        after = build(corrected, (1, 1), 10)
        fresh = record_probe(after, "p", 0, 0)
        self.assertEqual(fresh.squared, record.squared)
        self.assertNotEqual(fresh.witness, record.witness)
        self.assertEqual(fresh.witness.identity, 3)
        self.assertNotEqual(fresh.ticket, record.ticket)
        self.assertNotEqual(fresh.field.digest, record.field.digest)


class OwnershipTests(unittest.TestCase):
    def test_owner_boundary_is_half_open(self):
        frame = Frame(4, 4)
        region = OwnerRegion("s", 0, 4, 0, 4)
        self.assertTrue(region.contains(3, 3))
        self.assertFalse(region.contains(4, 3))
        self.assertFalse(region.contains(3, 4))

    def test_overlapping_owners_are_rejected(self):
        frame = Frame(4, 4)
        inventory = TileInventory()
        inventory.add(Tile("a", "s", 1, Box(0, 4, 0, 4), tuple((0,) * 4 for _ in range(4))))
        with self.assertRaises(ValueError):
            LabelMap(frame, [OwnerRegion("s", 0, 4, 0, 3),
                             OwnerRegion("t", 0, 4, 2, 4)], inventory)

    def test_unowned_cells_are_unknown(self):
        frame = Frame(4, 4)
        inventory = TileInventory()
        inventory.add(Tile("a", "s", 1, Box(0, 4, 0, 2), tuple((1,) * 2 for _ in range(4))))
        label_map = LabelMap(frame, [OwnerRegion("s", 0, 4, 0, 2)], inventory)
        self.assertEqual(label_map.label_at(0, 0), 1)
        self.assertEqual(label_map.label_at(0, 3), -1)
        self.assertIsNone(label_map.owner_of(0, 3))

    def test_gap_explained_by_context_but_not_defined(self):
        frame = Frame(2, 4)
        inventory = TileInventory()
        inventory.add(Tile("owner", "s", 1, Box(0, 2, 0, 2),
                           ((1, -1), (1, 1))))
        inventory.add(Tile("other", "t", 1, Box(0, 2, 1, 3),
                           ((0, 0), (0, 0))))
        label_map = LabelMap(frame, [OwnerRegion("s", 0, 2, 0, 2),
                                     OwnerRegion("t", 0, 2, 2, 4)], inventory)
        # 拥有者回传未知，另一来源的重叠读回不能改写它。
        self.assertEqual(label_map.label_at(0, 1), -1)
        explanation = label_map.explain_cell(0, 1)
        roles = {item["tile"]: item["role"] for item in explanation["holders"]}
        self.assertEqual(roles, {"owner": "owner", "other": "context"})
        self.assertTrue(explanation["context_only"])
        self.assertIn("不能定义前景", explanation["reason"])
        report = label_map.gap_report()
        self.assertEqual(report["unknown_cells"], 3)
        self.assertEqual(report["explained_by_context"], 1)

    def test_tile_outside_owner_scope_does_not_define(self):
        frame = Frame(2, 4)
        inventory = TileInventory()
        inventory.add(Tile("wide", "s", 1, Box(0, 2, 0, 4),
                           ((1, 0, 0, 0), (1, 1, 1, 1))))
        label_map = LabelMap(frame, [OwnerRegion("s", 0, 2, 0, 2)], inventory)
        self.assertEqual(label_map.label_at(0, 0), 1)
        self.assertEqual(label_map.label_at(0, 3), -1)
        holders = label_map.coverage(0, 3)
        self.assertEqual([item["role"] for item in holders], ["context"])


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.cols = (int(v) for v in CAPACITY["grid"])
        self.tile_rows, self.tile_cols = (int(v) for v in CAPACITY["tile_hw"])
        self.spacing = tuple(int(v) for v in CAPACITY["spacing"])
        self.radius = int(CAPACITY["radius"])
        self.budget = int(CAPACITY["peak_scratch_bytes"])

    def _sparse_inventory(self):
        rng = np.random.default_rng(int(CAPACITY["seed"]))
        inventory = TileInventory()
        for index, (row, col) in enumerate(
            [(0, 0), (0, self.tile_cols + 7), (self.tile_rows + 3, 0),
             (2000, 3000)]
        ):
            values = rng.integers(0, 2, size=(self.tile_rows, self.tile_cols),
                                  dtype=np.int8)
            values[:2, :2] = -1
            inventory.add(Tile(
                f"t{row}_{col}", "s", 1,
                Box(row, row + self.tile_rows, col, col + self.tile_cols),
                tuple(tuple(int(v) for v in line) for line in values),
            ))
        return inventory

    def test_streaming_stays_within_scratch_budget(self):
        frame = Frame(self.rows, self.cols)
        label_map = LabelMap(
            frame, [OwnerRegion("s", 0, self.rows, 0, self.cols)],
            self._sparse_inventory(),
        )
        field = SignedDistanceField(
            label_map, Calibration(1, Gauge(self.spacing, self.radius)),
            scratch_bytes=self.budget,
        )
        box = Box(2000, 2000 + 2 * self.tile_rows, 3000, 3000 + 2 * self.tile_cols)
        tracemalloc.start()
        try:
            for _tile, payload in field.iter_tiles(box):
                self.assertLessEqual(payload["status"].shape[0], self.tile_rows)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLessEqual(peak, self.budget)

    def test_no_full_frame_candidate_allocation(self):
        frame = Frame(self.rows, self.cols)
        label_map = LabelMap(
            frame, [OwnerRegion("s", 0, self.rows, 0, self.cols)],
            self._sparse_inventory(),
        )
        field = SignedDistanceField(
            label_map, Calibration(1, Gauge(self.spacing, self.radius)),
            scratch_bytes=self.budget,
        )
        # 只请求远处一个块；未触及的幅面不会被分配。
        box = Box(4000, 4000 + self.tile_rows, 4000, 4000 + self.tile_cols)
        tracemalloc.start()
        try:
            features = field.features(box)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(features["status"].shape, box.shape)
        self.assertLess(peak, self.budget * 3)
        self.assertEqual(features["status"].size, box.area)

    def test_automatic_tile_shape_fits_budget(self):
        frame = Frame(self.rows, self.cols)
        label_map = LabelMap(
            frame, [OwnerRegion("s", 0, self.rows, 0, self.cols)],
            self._sparse_inventory(),
        )
        field = SignedDistanceField(
            label_map, Calibration(1, Gauge(self.spacing, self.radius)),
            scratch_bytes=self.budget,
        )
        self.assertGreaterEqual(field.tile_shape[0], 1)
        self.assertLessEqual(field.tile_shape[0], 96)
        self.assertLessEqual(field.tile_shape[1], 64)

    def test_impossible_budget_is_rejected(self):
        frame = Frame(10, 10)
        inventory = TileInventory()
        inventory.add(Tile("t", "s", 1, Box(0, 10, 0, 10),
                           tuple((0,) * 10 for _ in range(10))))
        label_map = LabelMap(frame, [OwnerRegion("s", 0, 10, 0, 10)], inventory)
        with self.assertRaises(ValueError):
            SignedDistanceField(
                label_map, Calibration(1, Gauge((1, 1), 500)),
                scratch_bytes=1024,
            )


if __name__ == "__main__":
    unittest.main()
