"""距离场语义：量规、等距见证、截断与待判定。"""
import json
import random
from pathlib import Path
import unittest

import numpy as np

from gauge_distance.capture import build_capture, load_capture
from gauge_distance.field import (
    FieldStatus,
    SignedDistanceField,
    Uncertainty,
)
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

DATA = Path(__file__).resolve().parents[1] / "data"


def single_tile_field(labels, spacing, radius, *, tile_shape=None, scratch=1 << 30):
    labels = np.asarray(labels, dtype=np.int8)
    rows, cols = labels.shape
    frame = Frame(rows, cols)
    tile = Tile("all", "s", 1, Box(0, rows, 0, cols),
                tuple(tuple(int(v) for v in line) for line in labels))
    inventory = TileInventory()
    inventory.add(tile)
    label_map = LabelMap(frame, [OwnerRegion("s", 0, rows, 0, cols)], inventory)
    field = SignedDistanceField(
        label_map, Calibration(1, Gauge(spacing, radius)),
        tile_shape=(rows, cols) if tile_shape is None else tile_shape,
        scratch_bytes=scratch,
    )
    return field


def brute_nearest(labels, spacing, row, col):
    """完整枚举的最近异类格：返回 ``(平方距离, 坐标 ID)`` 或 ``None``。"""
    labels = np.asarray(labels)
    rows, cols = labels.shape
    label = int(labels[row, col])
    if label not in (0, 1):
        return None
    opposite = 0 if label == 1 else 1
    best = None
    for rr in range(rows):
        for cc in range(cols):
            if int(labels[rr, cc]) != opposite:
                continue
            dy = (row - rr) * spacing[0]
            dx = (col - cc) * spacing[1]
            squared = dy * dy + dx * dx
            identity = rr * cols + cc
            if best is None or squared < best[0] or (
                squared == best[0] and identity < best[1]
            ):
                best = (squared, identity)
    return best


class GaugeTests(unittest.TestCase):
    def test_centers_use_per_axis_spacing(self):
        gauge = Gauge((3, 2), 6)
        self.assertEqual(gauge.center(2, 3), (6, 6))
        self.assertEqual(gauge.center(1, 4), (3, 8))

    def test_cell_of_point_uses_centered_half_open_cells(self):
        gauge = Gauge((3, 2), 6)
        self.assertEqual(gauge.cell_of_point(3, 6), (1, 3))
        # 格 (0,2) 的中心单元为 y∈[-1.5,1.5)、x∈[3,5)。
        self.assertEqual(gauge.cell_of_point(1.4, 4.9), (0, 2))
        # 落在共有边界上时归入下（右）侧格。
        self.assertEqual(gauge.cell_of_point(0, 0), (0, 0))
        self.assertEqual(gauge.cell_of_point(1.5, 5.0), (1, 3))
        self.assertEqual(gauge.cell_of_point(6, 8), (2, 4))
        self.assertEqual(gauge.cell_of_point(-0.0, 0.0), (0, 0))

    def test_distance_is_not_isotropic(self):
        labels = [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
        field = single_tile_field(labels, (5, 1), 100)
        # (0,3) 的正下方就是背景 (1,3)：行差 1、间距 5。
        self.assertAlmostEqual(field.cell(0, 3).value, 5.0)
        # (1,2) 的右侧是背景：列差 1、间距 1。
        self.assertAlmostEqual(field.cell(1, 2).value, 1.0)
        # (1,0) 同时有行差 0、列差 3（3）与行差 1、列差 1 的候选；取较近者。
        self.assertAlmostEqual(field.cell(1, 0).value, 3.0)
        # (0,0) 行差 1、列差 3 → sqrt(25+9)。
        self.assertAlmostEqual(field.cell(0, 0).value, np.sqrt(34.0))


class NearestWitnessTests(unittest.TestCase):
    def test_equal_distance_breaks_tie_by_cell_identity(self):
        # 前景格 (1,1) 同时距背景 (0,1) 与 (1,0) 1 格：取行优先的更小 ID。
        labels = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        field = single_tile_field(labels, (1, 1), 10)
        result = field.cell(1, 1)
        self.assertEqual(result.status, FieldStatus.TRUSTED)
        self.assertEqual(result.squared, 1)
        self.assertEqual(result.witness.identity, 0 * 3 + 1)
        self.assertEqual((result.witness.row, result.witness.col), (0, 1))

    def test_grid_matches_brute_force_on_random_labels(self):
        rng = random.Random(2024)
        for _ in range(120):
            rows = rng.randint(1, 7)
            cols = rng.randint(1, 7)
            spacing = (rng.randint(1, 4), rng.randint(1, 4))
            radius = rng.randint(0, 12)
            labels = np.array(
                [[rng.choice([-1, 0, 1]) for _ in range(cols)] for _ in range(rows)],
                dtype=np.int8,
            )
            field = single_tile_field(labels, spacing, radius)
            out = field.arrays(Box(0, rows, 0, cols))
            cutoff = radius * radius
            for row in range(rows):
                for col in range(cols):
                    label = int(labels[row, col])
                    status = FieldStatus(int(out["status"][row, col]))
                    known = brute_nearest(labels, spacing, row, col)
                    known_in = known is not None and known[0] <= cutoff
                    unknown_in = False
                    unknown_key = None
                    for rr in range(rows):
                        for cc in range(cols):
                            if labels[rr, cc] != -1:
                                continue
                            dy = (row - rr) * spacing[0]
                            dx = (col - cc) * spacing[1]
                            squared = dy * dy + dx * dx
                            if squared > cutoff:
                                continue
                            unknown_in = True
                            key = (squared, rr * cols + cc)
                            if unknown_key is None or key < unknown_key:
                                unknown_key = key
                    if label == -1:
                        self.assertEqual(status, FieldStatus.NO_TARGET)
                    elif unknown_in and (
                        not known_in or unknown_key < known
                    ):
                        self.assertEqual(status, FieldStatus.UNDETERMINED)
                    elif known_in:
                        self.assertEqual(status, FieldStatus.TRUSTED)
                        self.assertEqual(int(out["squared"][row, col]), known[0])
                        self.assertEqual(
                            int(out["witness_row"][row, col]) * cols
                            + int(out["witness_col"][row, col]),
                            known[1],
                        )
                    elif unknown_in:
                        self.assertEqual(status, FieldStatus.UNDETERMINED)
                    else:
                        self.assertEqual(status, FieldStatus.TRUNCATED_NO_WITNESS)
                        self.assertEqual(
                            float(out["value"][row, col]),
                            float(radius if label == 1 else -radius),
                        )


class TruncationTests(unittest.TestCase):
    def test_candidate_exactly_at_radius_counts(self):
        labels = [[1, 1, 0]]
        field = single_tile_field(labels, (1, 2), 4)
        # 列差 2、间距 2 → 距离恰为 4。
        result = field.cell(0, 0)
        self.assertEqual(result.status, FieldStatus.TRUSTED)
        self.assertEqual(result.squared, 16)

    def test_beyond_radius_is_signed_threshold_without_witness(self):
        labels = [[1, 1, 1, 0]]
        field = single_tile_field(labels, (1, 1), 2)
        result = field.cell(0, 0)
        self.assertEqual(result.status, FieldStatus.TRUNCATED_NO_WITNESS)
        self.assertEqual(result.value, 2.0)
        self.assertIsNone(result.squared)
        self.assertIsNone(result.witness)
        self.assertFalse(result.eligible_for_training)

    def test_unknown_query_cell_has_no_target(self):
        labels = [[1, -1, 0]]
        field = single_tile_field(labels, (1, 1), 4)
        result = field.cell(0, 1)
        self.assertEqual(result.status, FieldStatus.NO_TARGET)
        self.assertIsNone(result.value)
        self.assertIsNone(result.witness)


class UncertaintyTests(unittest.TestCase):
    def test_unknown_could_be_closer(self):
        # 查询格 (0,0) 目前最近背景在 (0,3)；未知格 (0,1) 若成为背景会拉近。
        labels = [[1, -1, 1, 0]]
        field = single_tile_field(labels, (1, 1), 10)
        result = field.cell(0, 0)
        self.assertEqual(result.status, FieldStatus.UNDETERMINED)
        self.assertEqual(result.uncertainty, Uncertainty.UNKNOWN_COULD_BE_CLOSER)
        self.assertIsNone(result.value)
        self.assertIsNone(result.witness)

    def test_unknown_equal_distance_but_smaller_identity(self):
        # 查询格 (2,2) 的右侧背景 (2,3) 距离 1（ID 11）；上方未知格 (1,2)
        # 距离同为 1，但 ID 6 更小：见证不唯一。
        labels = [
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 0],
        ]
        field = single_tile_field(labels, (1, 1), 10)
        result = field.cell(2, 2)
        self.assertEqual(result.status, FieldStatus.UNDETERMINED)
        self.assertEqual(result.uncertainty, Uncertainty.UNKNOWN_WITNESS_TIE)

    def test_unknown_supplies_the_only_witness(self):
        labels = [
            [1, 1, -1],
            [1, 1, 1],
        ]
        field = single_tile_field(labels, (1, 1), 10)
        result = field.cell(0, 1)
        self.assertEqual(result.status, FieldStatus.UNDETERMINED)
        self.assertEqual(result.uncertainty,
                         Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS)

    def test_equal_distance_tie_against_unknown(self):
        # 查询格 (2,2) 的左邻未知 (2,1) 与其下方背景 (3,2) 等距 1，
        # 未知格的 ID 更小，因此见证不唯一。
        labels = [
            [1, 1, 1],
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, 0],
        ]
        field = single_tile_field(labels, (1, 1), 10)
        result = field.cell(2, 2)
        self.assertEqual(result.status, FieldStatus.UNDETERMINED)
        self.assertEqual(result.uncertainty, Uncertainty.UNKNOWN_WITNESS_TIE)

    def test_unknown_outside_radius_does_not_matter(self):
        labels = [[1, 1, 1, 0, -1, 1]]
        field = single_tile_field(labels, (1, 1), 2)
        result = field.cell(0, 0)
        # 背景 (0,3) 距离 3 超出半径 2；未知 (0,4) 也超出。
        self.assertEqual(result.status, FieldStatus.TRUNCATED_NO_WITNESS)
        self.assertEqual(result.value, 2.0)


class TilingTests(unittest.TestCase):
    def test_block_shape_does_not_change_results(self):
        rng = random.Random(99)
        for _ in range(30):
            rows = rng.randint(1, 18)
            cols = rng.randint(1, 18)
            spacing = (rng.randint(1, 4), rng.randint(1, 4))
            radius = rng.randint(0, 9)
            labels = np.array(
                [[rng.choice([-1, 0, 1]) for _ in range(cols)] for _ in range(rows)],
                dtype=np.int8,
            )
            reference = single_tile_field(labels, spacing, radius).arrays(
                Box(0, rows, 0, cols))
            for shape in [(1, 1), (3, 5), (7, 2), (rows, 1), (1, cols)]:
                field = single_tile_field(labels, spacing, radius, tile_shape=shape)
                out = field.arrays(Box(0, rows, 0, cols))
                for key in reference:
                    self.assertTrue(np.array_equal(reference[key], out[key]),
                                    f"{key} 随块形状 {shape} 变化")

    def test_window_subset_matches_full_frame(self):
        rng = random.Random(5)
        rows, cols = 16, 14
        labels = np.array(
            [[rng.choice([-1, 0, 1]) for _ in range(cols)] for _ in range(rows)],
            dtype=np.int8,
        )
        field = single_tile_field(labels, (2, 3), 7, tile_shape=(4, 4))
        full = field.arrays(Box(0, rows, 0, cols))
        sub = field.arrays(Box(3, 11, 2, 9))
        for key in full:
            self.assertTrue(np.array_equal(full[key][3:11, 2:9], sub[key]), key)

    def test_oversized_tile_shape_is_rejected(self):
        labels = np.zeros((2, 2), dtype=np.int8)
        with self.assertRaises(ValueError):
            single_tile_field(labels, (1, 1), 100, tile_shape=(4096, 4096),
                              scratch=4 << 20)

    def test_auto_tile_shape_respects_budget(self):
        labels = np.zeros((2, 2), dtype=np.int8)
        frame = Frame(2, 2)
        tile = Tile("all", "s", 1, Box(0, 2, 0, 2),
                    ((0, 0), (0, 0)))
        inventory = TileInventory()
        inventory.add(tile)
        label_map = LabelMap(frame, [OwnerRegion("s", 0, 2, 0, 2)], inventory)
        field = SignedDistanceField(
            label_map, Calibration(1, Gauge((6, 4), 100)),
            scratch_bytes=4 << 20,
        )
        self.assertLessEqual(field.tile_shape[0], 96)
        self.assertGreaterEqual(field.tile_shape[0], 1)


class CaptureProbeTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture(DATA / "capture.json")
        self.field = self.capture.field()

    def test_latest_owner_version_wins(self):
        self.assertEqual(self.field.credentials, (("left", 1), ("right", 2)))

    def test_probes_use_actual_position(self):
        field = self.field
        centered = field.probe(3, 6)
        self.assertEqual(centered.status, FieldStatus.TRUSTED)
        # (3,6) 属于格 (1,3)，其中心恰为 (3,6)。
        self.assertEqual((centered.row, centered.col), (1, 3))
        self.assertAlmostEqual(centered.value, np.sqrt(13))

        # (4,7) 在格 (1,4) 的中心单元内，但距离用实际坐标而不是格点中心。
        offset = field.probe(4, 7)
        self.assertEqual((offset.row, offset.col), (1, 4))
        self.assertAlmostEqual(offset.value, 5.0)

        # (3,7) 落在格 (1,3)/(1,4) 的共有列边界上，归入右侧格 (1,4)。
        boundary = field.probe(3, 7)
        self.assertEqual((boundary.row, boundary.col), (1, 4))
        self.assertAlmostEqual(boundary.value, np.sqrt(18))

    def test_probe_is_not_shared_with_cell_center(self):
        centered = self.field.probe(3, 6)
        offset = self.field.probe(4, 7)
        self.assertNotEqual((centered.row, centered.col),
                            (offset.row, offset.col))
        self.assertNotEqual(centered.squared, offset.squared)
        # 同一格的格点中心见证 (0,2) 不被整格共用：见证坐标保持一致，
        # 但距离按各自定位点计算。
        self.assertEqual(centered.witness, offset.witness)
        self.assertNotEqual(centered.value, offset.value)

    def test_offset_does_not_change_grid_or_area(self):
        offset = self.field.probe(4, 7)
        gauge = self.field.gauge
        self.assertEqual(gauge.cell_of_point(4, 7), (1, 4))
        self.assertEqual(gauge.center(1, 4), (3, 8))
        self.assertEqual(gauge.spacing, (3, 2))

    def test_probe_works_for_background_cells(self):
        # (0,6) 属于格 (0,3)，是前景；用物理坐标 (0,0) 落在背景格 (0,0)。
        result = self.field.probe(0, 6)
        self.assertEqual((result.row, result.col), (0, 3))
        self.assertEqual(result.sign, 1)
        background = self.field.probe(0, 0)
        self.assertEqual((background.row, background.col), (0, 0))
        self.assertEqual(background.label, 0)
        self.assertEqual(background.sign, -1)
        self.assertEqual(background.status, FieldStatus.TRUSTED)
        self.assertEqual((background.witness.row, background.witness.col), (1, 1))

    def test_prior_version_can_be_requeried(self):
        prior = self.capture.field(selection=self.capture.prior_selection())
        self.assertEqual(prior.credentials, (("left", 1), ("right", 1)))
        latest_probe = self.field.probe(3, 7)
        prior_probe = prior.probe(3, 7)
        self.assertNotEqual(latest_probe.squared, prior_probe.squared)

    def test_gauge_change_changes_distances(self):
        changed = self.capture.field(calibration=self.capture.gauge_change)
        self.assertEqual(changed.gauge.spacing, (2, 2))
        self.assertEqual(changed.gauge.radius, 5)
        self.assertAlmostEqual(changed.probe(4, 7).value, np.sqrt(17))

    def test_context_never_defines_owner_foreground(self):
        # 右瓦片从列 3 起，越出 lit 的管辖范围（列 4 起）。
        labels = self.field.labels
        explanation = labels.explain_cell(0, 3)
        self.assertEqual(explanation["owner"], "slide")
        self.assertEqual(explanation["label"],
                         self.capture.inventory.get("left", 1).label_at(0, 3))
        holders = {item["tile"]: item for item in explanation["holders"]}
        self.assertEqual(holders["left"]["role"], "owner")
        self.assertEqual(holders["right"]["role"], "context")
        self.assertTrue(explanation["defined_by_owner"])

    def test_unreturned_owner_cell_stays_unknown(self):
        # 收窄 lit 的瓦片到它自己的管辖范围 (列 4 起)，让 slide 的管辖格 (0,3)
        # 不再有 lit 的重叠读回；拥有者未回传的格不得被当作背景。
        capture = load_capture(DATA / "capture.json")
        original = capture.inventory.get("right", 2)
        narrow = Tile("right", "lit", 3, Box(0, 5, 4, 8),
                      tuple(tuple(int(v) for v in line[1:])
                            for line in original.values))
        capture.inventory.add(narrow)
        field = capture.field(selection={"left": 1, "right": 3})
        self.assertEqual(field.credentials, (("left", 1), ("right", 3)))
        # slide 拥有 (0,3)，值为 1；lit 的重叠读回不能把它改写成 0。
        self.assertEqual(field.labels.label_at(0, 3), 1)
        # lit 缩小后，(0,3) 不再有上下文覆盖。
        explanation = field.labels.explain_cell(0, 3)
        self.assertEqual([item["tile"] for item in explanation["holders"]], ["left"])


if __name__ == "__main__":
    unittest.main()
