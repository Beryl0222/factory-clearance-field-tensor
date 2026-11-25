"""训练记录：可证实性、瓦片凭据、换版与追查。"""
from pathlib import Path
import unittest

import numpy as np

from gauge_distance.capture import load_capture
from gauge_distance.field import FieldStatus, Uncertainty, reason_for
from gauge_distance.model import Box, Calibration, Gauge, Tile
from gauge_distance.training import (
    TrainingError,
    TrainingLedger,
    assemble_sample,
    evaluate_probe,
    field_for_credentials,
    field_version,
    gauge_of,
    record_probe,
    verify_record,
)

DATA = Path(__file__).resolve().parents[1] / "data"


class TrainingRecordTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture(DATA / "capture.json")
        self.field = self.capture.field()

    def test_trusted_probe_is_training_eligible(self):
        record = record_probe(self.field, "p-center", 3, 6)
        self.assertTrue(record.eligible)
        self.assertEqual(record.status, FieldStatus.TRUSTED)
        self.assertIsNotNone(record.witness)
        self.assertAlmostEqual(record.value, np.sqrt(13))
        self.assertEqual(record.cell.row, 1)
        self.assertEqual(record.cell.col, 3)
        self.assertAlmostEqual(gauge_of(record).spacing[0], 3)

    def test_record_ticket_is_stable_for_same_field(self):
        first = record_probe(self.field, "p-center", 3, 6)
        again = record_probe(self.capture.field(), "p-center", 3, 6)
        self.assertEqual(first.ticket, again.ticket)
        self.assertEqual(first.field.digest, again.field.digest)

    def test_untrusted_probe_is_rejected_for_training(self):
        # 数据里的定位点都可证实；用同量规下的截断格确认训练会拒绝无见证目标。
        far = self.capture.field(calibration=Calibration(1, Gauge((3, 2), 0)))
        record = evaluate_probe(far, "p-center", 3, 6)
        self.assertEqual(record.status, FieldStatus.TRUNCATED_NO_WITNESS)
        self.assertFalse(record.eligible)
        self.assertIn("截断", record.reason)
        with self.assertRaises(TrainingError):
            record_probe(far, "p-center", 3, 6)

    def test_no_target_probe_is_rejected(self):
        # 数据里 (0,3) 的两块瓦片都给出确定值；用未知格所在的定位点检查拒绝。
        capture = load_capture(DATA / "capture.json")
        narrow = Tile("right", "lit", 3, Box(0, 5, 4, 8),
                      tuple(tuple(int(v) for v in line[1:])
                            for line in capture.inventory.get("right", 2).values))
        capture.inventory.add(narrow)
        # 把 left 也收窄，使 (0,3) 成为无人回传的格。
        left = capture.inventory.get("left", 1)
        shortened = Tile("left", "slide", 2, Box(0, 5, 0, 3),
                         tuple(tuple(int(v) for v in line[:3])
                               for line in left.values))
        capture.inventory.add(shortened)
        field = capture.field(selection={"left": 2, "right": 3})
        record = evaluate_probe(field, "gap", 0, 6)
        self.assertEqual(record.status, FieldStatus.NO_TARGET)
        self.assertFalse(record.eligible)
        with self.assertRaises(TrainingError):
            record_probe(field, "gap", 0, 6)

    def test_verify_rejects_changed_field(self):
        record = record_probe(self.field, "p-center", 3, 6)
        prior = self.capture.field(selection=self.capture.prior_selection())
        with self.assertRaises(TrainingError):
            verify_record(record, prior)

    def test_verify_rejects_gauge_mix(self):
        record = record_probe(self.field, "p-center", 3, 6)
        changed = self.capture.field(calibration=self.capture.gauge_change)
        with self.assertRaises(TrainingError):
            verify_record(record, changed)

    def test_verify_accepts_same_content(self):
        record = record_probe(self.field, "p-center", 3, 6)
        self.assertEqual(verify_record(record, self.capture.field()), record)

    def test_verify_rejects_value_change_with_same_tile_credentials(self):
        # 同版本号但内容不同的瓦片被拒绝入库，因此不可能出现“同样凭据、不同内容”
        # 的距离场；再核对内容摘要确实变化时记录会被拒。
        capture = load_capture(DATA / "capture.json")
        original = capture.inventory.get("left", 1)
        changed_values = np.array(original.values)
        changed_values[0, 2] = 1  # 原为 0
        with self.assertRaises(ValueError):
            capture.inventory.add(Tile("left", "slide", 1, original.box,
                                       tuple(tuple(int(v) for v in line)
                                             for line in changed_values)))
        self.assertEqual(capture.inventory.get("left", 1), original)
        # 以新版本号加入后，旧记录因内容摘要不同必须换版。
        capture.inventory.add(Tile("left", "slide", 2, original.box,
                                   tuple(tuple(int(v) for v in line)
                                         for line in changed_values)))
        record = record_probe(self.field, "p-center", 3, 6)
        updated = capture.field()
        self.assertNotEqual(updated.credentials, record.credentials)
        with self.assertRaises(TrainingError):
            verify_record(record, updated)

    def test_content_digest_changes_when_witness_invalidated(self):
        # 距离值不变但见证失效：把 (0,2) 改成 -1 后，p-center 的最大距离见证
        # 仍是同一格时内容不变；这里直接验证瓦片内容变化一定改变版本摘要。
        capture = load_capture(DATA / "capture.json")
        original = capture.inventory.get("left", 1)
        changed_values = np.array(original.values)
        changed_values[4, 4] = -1  # 远处背景格变未知，距离与见证可能不变
        capture.inventory.add(Tile("left", "slide", 2, original.box,
                                   tuple(tuple(int(v) for v in line)
                                         for line in changed_values)))
        before = field_version(self.field)
        after = field_version(capture.field())
        self.assertNotEqual(before.digest, after.digest)

    def test_assemble_sample_rejects_mixed_gauges(self):
        centered = record_probe(self.field, "p-center", 3, 6)
        changed_field = self.capture.field(calibration=self.capture.gauge_change)
        changed = record_probe(changed_field, "p-center", 3, 6)
        with self.assertRaises(TrainingError):
            assemble_sample([centered, changed])

    def test_assemble_sample_rejects_ineligible(self):
        record = record_probe(self.field, "p-center", 3, 6)
        # 构造一条截断无见证记录（同量规、无见证）并拒绝。
        from gauge_distance.training import TrainingRecord
        from gauge_distance.model import Coord
        from dataclasses import replace
        fake = replace(
            record,
            status=FieldStatus.TRUNCATED_NO_WITNESS,
            value=float(record.field.radius),
            squared=None,
            witness=None,
        )
        with self.assertRaises(TrainingError):
            assemble_sample([record, fake])

    def test_sample_targets_include_witness_coordinates(self):
        records = [
            record_probe(self.field, "p-center", 3, 6),
            record_probe(self.field, "p-near", 2, 5),
        ]
        sample = assemble_sample(records)
        targets = sample.targets()
        self.assertEqual(len(targets), 2)
        self.assertEqual(set(sample.credentials), {("left", 1), ("right", 2)})
        self.assertEqual(targets[0][3], (0, 2))
        self.assertEqual(targets[1][0], 2)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.capture = load_capture(DATA / "capture.json")
        self.field = self.capture.field()
        self.ledger = TrainingLedger()
        self.ledger.issue(self.field, self.capture.probes)

    def test_issue_registers_current_records(self):
        for probe_id in self.capture.probes:
            current = self.ledger.current(probe_id)
            self.assertIsNotNone(current)
            self.assertTrue(current.eligible)
            self.assertEqual(len(self.ledger.history_of(probe_id)), 1)

    def test_reissue_only_for_changed_labels_inside_influence(self):
        changed = self.capture.field(selection=self.capture.prior_selection())
        report = self.ledger.reissue(self.field, changed, self.capture.probes)
        self.assertGreater(report["reissued"], 0)
        self.assertEqual(report["lost"], 0)
        self.assertEqual(report["new"], 0)
        # p-near 在两种版本下距离与见证都不变，只保留旧版。
        self.assertEqual(len(self.ledger.history_of("p-near")), 1)
        self.assertGreaterEqual(len(self.ledger.history_of("p-center")), 2)

    def test_unchanged_field_keeps_every_record(self):
        report = self.ledger.reissue(self.field, self.capture.field(),
                                     self.capture.probes)
        self.assertEqual(report["reissued"], 0)
        self.assertEqual(report["kept_old"], 0)
        self.assertEqual(report["unchanged"], len(self.capture.probes))
        for probe_id in self.capture.probes:
            self.assertEqual(len(self.ledger.history_of(probe_id)), 1)

    def test_old_credentials_still_queryable_after_reissue(self):
        changed = self.capture.field(selection=self.capture.prior_selection())
        old_credentials = self.field.credentials
        delivered = self.ledger.records_with_credentials(old_credentials)
        self.assertEqual(len(delivered), len(self.capture.probes))
        self.ledger.reissue(self.field, changed, self.capture.probes)
        # 旧记录还在台账里，可按旧凭据回查。
        still = self.ledger.records_with_credentials(old_credentials)
        self.assertEqual(len(still), len(delivered))
        for record in still:
            current = self.ledger.current(record.probe_id)
            if current.ticket != record.ticket:
                self.assertIsNotNone(record.superseded_by)

    def test_lost_eligibility_is_recorded_with_reason(self):
        # 用一个更小的半径让 p-center 成为截断无见证（半径 0）。
        tiny = self.capture.field(calibration=Calibration(1, Gauge((3, 2), 0)))
        ledger = TrainingLedger()
        ledger.issue(self.field, self.capture.probes)
        report = ledger.reissue(self.field, tiny, self.capture.probes)
        self.assertGreater(report["lost"], 0)
        current = ledger.current("p-center")
        self.assertFalse(current.eligible)
        explanation = ledger.explain("p-center")
        actions = [item["action"] for item in explanation["entries"]]
        self.assertIn("lost", actions)
        lost_entry = next(item for item in explanation["entries"]
                          if item["action"] == "lost")
        self.assertIn("截断", lost_entry["reason"])

    def test_explain_traces_kept_old_reason(self):
        # 影响域只在右半幅（右瓦片版本变化）；左侧定位点应保留旧版并说明原因。
        capture = load_capture(DATA / "capture.json")
        original = capture.inventory.get("left", 1)
        values = np.array(original.values)
        values[4, 4] = -1
        capture.inventory.add(Tile("left", "slide", 2, original.box,
                                   tuple(tuple(int(v) for v in line)
                                         for line in values)))
        updated = capture.field()
        self.ledger.reissue(self.field, updated, self.capture.probes)
        explanation = self.ledger.explain("p-near")
        kept = [item for item in explanation["entries"]
                if item["action"] == "kept_old"]
        self.assertTrue(kept)
        self.assertIn("保留旧版", kept[-1]["reason"])

    def test_field_for_credentials_rebuilds_old_view(self):
        delivered = self.ledger.records_with_credentials(self.field.credentials)
        rebuilt = field_for_credentials(
            self.capture.inventory, self.capture.owners, self.capture.frame,
            self.capture.calibration, self.field.credentials,
            tile_shape=(5, 8),
        )
        self.assertEqual(rebuilt.credentials, self.field.credentials)
        for record in delivered:
            verified = verify_record(record, rebuilt)
            self.assertEqual(verified.ticket, record.ticket)

    def test_field_version_includes_content_digest(self):
        version = field_version(self.field)
        self.assertEqual(version.credentials, self.field.credentials)
        self.assertEqual(len(version.digest), 64)
        self.assertNotEqual(version.digest,
                            field_version(self.capture.field(
                                selection=self.capture.prior_selection())).digest)

    def test_sample_uses_current_eligible_records(self):
        sample = self.ledger.sample()
        self.assertEqual(len(sample.records), len(self.capture.probes))
        self.assertTrue(all(record.eligible for record in sample.records))
        for record in sample.records:
            self.assertIs(record.status, FieldStatus.TRUSTED)
            self.assertIsNotNone(record.witness)

    def test_sample_rejects_missing_probe(self):
        with self.assertRaises(TrainingError):
            self.ledger.sample(["nope"])

    def test_reissue_when_witness_changed_but_distance_same(self):
        # p-center 的格 (1,3) 到背景 (0,2) 距离 sqrt(13)。擦除 (0,2)、在等距的
        # (2,2) 补入背景：距离数值不变，但见证换到 ID 更大的 (2,2)。
        capture = load_capture(DATA / "capture.json")
        original = capture.inventory.get("left", 1)
        values = np.array(original.values)
        self.assertEqual(int(values[0, 2]), 0)
        self.assertEqual(int(values[2, 2]), 1)
        values[0, 2] = 1
        values[2, 2] = 0
        capture.inventory.add(Tile("left", "slide", 2, original.box,
                                   tuple(tuple(int(v) for v in line)
                                         for line in values)))
        updated = capture.field()
        old_record = self.ledger.current("p-center")
        self.ledger.reissue(self.field, updated, self.capture.probes)
        current = self.ledger.current("p-center")
        self.assertNotEqual(current.ticket, old_record.ticket)
        self.assertEqual(current.squared, old_record.squared)
        self.assertNotEqual(current.witness, old_record.witness)
        self.assertEqual((current.witness.row, current.witness.col), (2, 2))
        superseded = self.ledger.record(old_record.ticket)
        self.assertEqual(superseded.superseded_by, current.ticket)
        self.assertNotEqual(current.field.digest, old_record.field.digest)


class ReasonTests(unittest.TestCase):
    def test_reasons_are_explicit_for_each_status(self):
        self.assertIn("未知", reason_for(FieldStatus.NO_TARGET))
        self.assertIn("截断", reason_for(FieldStatus.TRUNCATED_NO_WITNESS))
        self.assertIn("未知格", reason_for(
            FieldStatus.UNDETERMINED, Uncertainty.UNKNOWN_COULD_BE_CLOSER))
        self.assertIn("见证", reason_for(
            FieldStatus.UNDETERMINED, Uncertainty.UNKNOWN_WITNESS_TIE))
        self.assertIn("见证", reason_for(
            FieldStatus.UNDETERMINED,
            Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS))


if __name__ == "__main__":
    unittest.main()
