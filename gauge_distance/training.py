"""训练记录、瓦片凭据与版本追查。

每条训练记录都绑定到具体的**距离场版本**：量规（间距、半径）、瓦片凭据（``id``
与版本）以及瓦片内容的摘要。因此:

* 距离数值不变但见证失效时，凭据摘要变化，记录换版；
* 旧版记录仍可按旧的瓦片凭据回查；
* 采集员擦除或补入小块数据后，只有截断影响域内标签确实变化的定位点才重发，
  其余定位点保留旧版并记录原因。
* 一个训练样本不能把旧距离场的瓦片与新量规拼在一起。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from .field import CellResult, FieldStatus, SignedDistanceField, Uncertainty
from .model import (
    Box,
    Calibration,
    Coord,
    Gauge,
    LabelMap,
    Tile,
)


class TrainingError(RuntimeError):
    """训练记录不可证实、凭据不符或量规混用。"""


@dataclass(frozen=True)
class FieldVersion:
    """距离场版本：由量规、瓦片凭据与瓦片内容共同确定。"""

    calibration: int
    spacing: tuple[int, int]
    radius: int
    credentials: tuple[tuple[str, int], ...]
    tile_digest: str = ""
    digest: str = ""

    def as_dict(self) -> dict:
        return {
            "calibration": self.calibration,
            "spacing": list(self.spacing),
            "radius": self.radius,
            "credentials": [list(item) for item in self.credentials],
            "tile_digest": self.tile_digest,
            "digest": self.digest,
        }


def _digest(payload: Mapping) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tile_digest(tiles: Sequence[Tile]) -> str:
    parts = []
    for tile in sorted(tiles, key=lambda item: item.credential):
        parts.append({
            "id": tile.id,
            "source": tile.source,
            "version": tile.version,
            "box": tile.box.as_record(),
            "values": [list(line) for line in tile.values],
        })
    return _digest({"tiles": parts})


def field_version(field: SignedDistanceField) -> FieldVersion:
    """导出距离场版本；同一量规与同一瓦片内容得到同一摘要。"""
    gauge = field.gauge
    calibration = field.calibration.version
    credentials = field.credentials
    digest_tiles = _tile_digest(field.labels.tiles)
    payload = {
        "calibration": calibration,
        "spacing": list(gauge.spacing),
        "radius": gauge.radius,
        "credentials": [list(item) for item in credentials],
        "tile_digest": digest_tiles,
    }
    return FieldVersion(
        calibration=calibration,
        spacing=gauge.spacing,
        radius=gauge.radius,
        credentials=credentials,
        tile_digest=digest_tiles,
        digest=_digest(payload),
    )


@dataclass(frozen=True)
class TrainingRecord:
    """一条定位点的训练记录（可证实或已失去资格）。"""

    probe_id: str
    probe_y: float
    probe_x: float
    field: FieldVersion
    cell: Coord
    label: int
    sign: int
    status: FieldStatus
    squared: float | int | None
    value: float | None
    witness: Coord | None
    uncertainty: Uncertainty = Uncertainty.NONE
    reason: str = ""
    ticket: str = ""
    superseded_by: str | None = None

    @property
    def eligible(self) -> bool:
        """训练只接受可证实的符号距及坐标见证。"""
        return (
            self.status is FieldStatus.TRUSTED
            and self.witness is not None
            and self.squared is not None
            and self.value is not None
            and self.sign in (1, -1)
        )

    @property
    def calibration(self) -> int:
        return self.field.calibration

    @property
    def credentials(self) -> tuple[tuple[str, int], ...]:
        return self.field.credentials

    def as_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "probe": [self.probe_y, self.probe_x],
            "field": self.field.as_dict(),
            "cell": [self.cell.row, self.cell.col, self.cell.identity],
            "label": self.label,
            "sign": self.sign,
            "status": self.status.name,
            "squared": self.squared,
            "value": self.value,
            "witness": None if self.witness is None else [
                self.witness.row, self.witness.col, self.witness.identity
            ],
            "uncertainty": self.uncertainty.name,
            "reason": self.reason,
            "ticket": self.ticket,
            "eligible": self.eligible,
            "superseded_by": self.superseded_by,
        }


def _ticket(record: TrainingRecord) -> str:
    payload = {
        "probe_id": record.probe_id,
        "probe": [record.probe_y, record.probe_x],
        "field": record.field.digest,
        "cell": [record.cell.row, record.cell.col, record.cell.identity],
        "label": record.label,
        "sign": record.sign,
        "status": int(record.status),
        "squared": record.squared,
        "value": record.value,
        "witness": None if record.witness is None else [
            record.witness.row, record.witness.col, record.witness.identity
        ],
        "uncertainty": int(record.uncertainty),
    }
    return _digest(payload)


def _same_content(left: TrainingRecord, right: TrainingRecord) -> bool:
    """判断两条记录的标签内容是否相同（不含瓦片凭据本身）。"""
    return (
        left.calibration == right.calibration
        and left.field.spacing == right.field.spacing
        and left.field.radius == right.field.radius
        and left.cell == right.cell
        and left.label == right.label
        and left.sign == right.sign
        and left.status is right.status
        and left.squared == right.squared
        and left.value == right.value
        and left.witness == right.witness
        and left.uncertainty is right.uncertainty
    )


def _make_record(field: SignedDistanceField, probe_id: str, y: float, x: float,
                 result: CellResult, reason: str = "") -> TrainingRecord:
    version = field_version(field)
    record = TrainingRecord(
        probe_id=probe_id,
        probe_y=y,
        probe_x=x,
        field=version,
        cell=Coord(result.row, result.col, result.identity),
        label=result.label,
        sign=result.sign,
        status=result.status,
        squared=result.squared,
        value=result.value,
        witness=result.witness,
        uncertainty=result.uncertainty,
        reason=reason or result.reason,
    )
    return replace(record, ticket=_ticket(record))


def evaluate_probe(field: SignedDistanceField, probe_id: str, y: float, x: float) -> TrainingRecord:
    """计算定位点的记录；不可证实的结果也返回，用 ``eligible`` 与原因说明。"""
    result = field.probe(y, x)
    reason = result.reason
    if result.status is FieldStatus.TRUNCATED_NO_WITNESS:
        reason = f"{reason}；截断域外无见证，不能用作训练目标"
    elif result.status is FieldStatus.UNDETERMINED:
        reason = f"{reason}；待判定区域，不能用作训练目标"
    elif result.status is FieldStatus.NO_TARGET:
        reason = f"{reason}；该定位点所在格没有符号距"
    return _make_record(field, probe_id, y, x, result, reason)


def record_probe(field: SignedDistanceField, probe_id: str, y: float, x: float) -> TrainingRecord:
    """生成可证实的训练记录；不可证实则拒绝。"""
    record = evaluate_probe(field, probe_id, y, x)
    if not record.eligible:
        raise TrainingError(
            f"定位点 {probe_id} 不可用于训练：{record.reason}"
        )
    return record


def verify_record(record: TrainingRecord, field: SignedDistanceField) -> TrainingRecord:
    """按给定距离场复核记录；失败时抛出 ``TrainingError`` 说明原因。"""
    version = field_version(field)
    if version.digest != record.field.digest:
        if version.calibration != record.field.calibration or version.spacing != record.field.spacing \
                or version.radius != record.field.radius:
            raise TrainingError(
                f"定位点 {record.probe_id} 的量规已变更，旧距离场不能与新量规拼用"
            )
        raise TrainingError(
            f"定位点 {record.probe_id} 的瓦片凭据或内容与当前距离场不符，"
            "请按旧凭据回查或重新签发"
        )
    if not record.eligible:
        raise TrainingError(f"定位点 {record.probe_id} 不是可证实的训练目标：{record.reason}")
    fresh = evaluate_probe(field, record.probe_id, record.probe_y, record.probe_x)
    if not fresh.eligible:
        raise TrainingError(f"定位点 {record.probe_id} 在新距离场中已不可证实：{fresh.reason}")
    if not _same_content(record, fresh):
        raise TrainingError(
            f"定位点 {record.probe_id} 的距离或见证已变化，旧记录必须换版"
        )
    return record


def assemble_sample(records: Iterable[TrainingRecord],
                    calibration: Calibration | None = None) -> "TrainingSample":
    """把记录组装成训练样本；拒绝失去资格的记录与新旧量规混用。"""
    items = list(records)
    if not items:
        raise TrainingError("训练样本为空")
    if calibration is None:
        version = items[0].calibration
        spacing = items[0].field.spacing
        radius = items[0].field.radius
    else:
        version = calibration.version
        spacing = calibration.spacing
        radius = calibration.radius
    for record in items:
        if not record.eligible:
            raise TrainingError(f"定位点 {record.probe_id} 不满足训练条件：{record.reason}")
        if record.calibration != version or record.field.spacing != spacing \
                or record.field.radius != radius:
            raise TrainingError(
                f"定位点 {record.probe_id} 来自量规 {record.calibration}"
                f"{record.field.spacing}@{record.field.radius}，"
                f"不能与量规 {version}{spacing}@{radius} 拼在一个训练样本里"
            )
    return TrainingSample(tuple(items), version, spacing, radius)


@dataclass(frozen=True)
class TrainingSample:
    records: tuple[TrainingRecord, ...]
    calibration: int
    spacing: tuple[int, int]
    radius: int

    @property
    def credentials(self) -> tuple[tuple[str, int], ...]:
        items: set[tuple[str, int]] = set()
        for item in self.records:
            items.update(item.credentials)
        return tuple(sorted(items))

    def targets(self) -> list[tuple[float, float, float, tuple[int, int]]]:
        """``(probe_y, probe_x, signed_distance, (witness_row, witness_col))``。"""
        return [
            (r.probe_y, r.probe_x, r.value, (r.witness.row, r.witness.col))
            for r in self.records
        ]


# ---------------------------------------------------------------------------
# 版本台账
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    probe_id: str
    action: str
    reason: str
    ticket: str
    field_digest: str
    calibration: int
    credentials: tuple[tuple[str, int], ...]
    previous_ticket: str | None = None

    def as_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "action": self.action,
            "reason": self.reason,
            "ticket": self.ticket,
            "field_digest": self.field_digest,
            "calibration": self.calibration,
            "credentials": [list(item) for item in self.credentials],
            "previous_ticket": self.previous_ticket,
        }


@dataclass
class TrainingLedger:
    """按定位点保存历次签发，支持按旧凭据回查与追查原因。"""

    records: dict[str, TrainingRecord] = field(default_factory=dict)
    history: dict[str, list[str]] = field(default_factory=dict)
    entries: list[LedgerEntry] = field(default_factory=list)

    # -- 基本访问 ---------------------------------------------------------

    def current(self, probe_id: str) -> TrainingRecord | None:
        ticket = self._current_ticket(probe_id)
        if ticket is None:
            return None
        return self.records[ticket]

    def record(self, ticket: str) -> TrainingRecord:
        return self.records[ticket]

    def history_of(self, probe_id: str) -> list[TrainingRecord]:
        return [self.records[ticket] for ticket in self.history.get(probe_id, ())]

    def probe_ids(self) -> list[str]:
        return sorted(self.history)

    def entries_for(self, probe_id: str) -> list[LedgerEntry]:
        return [item for item in self.entries if item.probe_id == probe_id]

    def explain(self, probe_id: str) -> dict:
        """追查一条训练记录的当前版本与每次变动的理由。"""
        current = self.current(probe_id)
        return {
            "probe_id": probe_id,
            "current": None if current is None else current.as_dict(),
            "history": [record.as_dict() for record in self.history_of(probe_id)],
            "entries": [item.as_dict() for item in self.entries_for(probe_id)],
        }

    def records_with_credentials(self, credentials: Iterable[tuple[str, int]]) -> list[TrainingRecord]:
        """按旧瓦片凭据回查已交付的记录。"""
        wanted = tuple(sorted((str(i), int(v)) for i, v in credentials))
        return sorted(
            (record for record in self.records.values() if record.credentials == wanted),
            key=lambda item: (item.probe_id, item.ticket),
        )

    def current_records(self) -> list[TrainingRecord]:
        """每个定位点的当前记录；失去资格的记录也在其中，用 ``eligible`` 区分。"""
        result = []
        for probe_id in self.probe_ids():
            record = self.current(probe_id)
            if record is not None:
                result.append(record)
        return result

    def eligible_records(self) -> list[TrainingRecord]:
        """当前可用于训练的定位点记录。"""
        return [record for record in self.current_records() if record.eligible]

    def sample(self, probe_ids: Iterable[str] | None = None,
               calibration: Calibration | None = None) -> "TrainingSample":
        """按当前记录组装训练样本；默认取全部可证实记录。

        记录来自不同量规版本时由 :func:`assemble_sample` 拒绝，从而不会把旧距离
        场的瓦片与新量规拼在一起。
        """
        if probe_ids is None:
            records = self.eligible_records()
        else:
            records = []
            for probe_id in probe_ids:
                record = self.current(probe_id)
                if record is None:
                    raise TrainingError(f"定位点 {probe_id} 没有登记记录")
                records.append(record)
        return assemble_sample(records, calibration)

    def _current_ticket(self, probe_id: str) -> str | None:
        chain = self.history.get(probe_id)
        if not chain:
            return None
        return chain[-1]

    def _append(self, record: TrainingRecord, previous: str | None) -> None:
        self.records[record.ticket] = record
        self.history.setdefault(record.probe_id, []).append(record.ticket)
        if previous is not None and previous in self.records:
            self.records[previous] = replace(self.records[previous],
                                             superseded_by=record.ticket)

    def _log(self, record: TrainingRecord, action: str, reason: str,
             previous: str | None = None) -> None:
        self.entries.append(LedgerEntry(
            probe_id=record.probe_id,
            action=action,
            reason=reason,
            ticket=record.ticket,
            field_digest=record.field.digest,
            calibration=record.calibration,
            credentials=record.credentials,
            previous_ticket=previous,
        ))

    # -- 签发与刷新 -------------------------------------------------------

    def issue(self, field: SignedDistanceField,
              probes: Mapping[str, tuple[float, float]]) -> list[TrainingRecord]:
        """首次为定位点签发记录（不可证实的也会登记，供追查）。"""
        issued = []
        for probe_id, (y, x) in probes.items():
            record = evaluate_probe(field, probe_id, y, x)
            self._append(record, None)
            action = "issued" if record.eligible else "ineligible"
            self._log(record, action, record.reason)
            issued.append(record)
        return issued

    def reissue(self, old_field: SignedDistanceField, new_field: SignedDistanceField,
                probes: Mapping[str, tuple[float, float]] | None = None) -> dict:
        """瓦片或量规更新后，只为截断影响域内标签变化的定位点重发。

        ``probes`` 未给出时沿用台账里当前记录的定位坐标。返回各类计数与条目。
        """
        old_version = field_version(old_field)
        new_version = field_version(new_field)
        changed_all = (
            old_version.calibration != new_version.calibration
            or old_version.spacing != new_version.spacing
            or old_version.radius != new_version.radius
        )
        identical = old_version.digest == new_version.digest
        boxes = [] if identical else (
            [new_field.frame.box] if changed_all
            else self._influence_boxes(old_field, new_field)
        )

        points = dict(probes) if probes is not None else {
            probe_id: (record.probe_y, record.probe_x)
            for probe_id in self.probe_ids()
            if (record := self.current(probe_id)) is not None
        }
        report = {"kept_old": 0, "reissued": 0, "lost": 0, "new": 0, "unchanged": 0,
                  "entries": []}
        for probe_id, (y, x) in points.items():
            previous = self.current(probe_id)
            fresh = evaluate_probe(new_field, probe_id, y, x)
            if previous is None:
                self._append(fresh, None)
                self._log(fresh, "issued" if fresh.eligible else "ineligible", fresh.reason)
                report["new"] += 1
                report["entries"].append(probe_id)
                continue
            if identical:
                self._log(previous, "unchanged", "距离场版本未变，保留旧版")
                report["unchanged"] += 1
                report["entries"].append(probe_id)
                continue
            inside = any(box.contains(previous.cell.row, previous.cell.col)
                         for box in boxes)
            if not inside:
                self._log(previous, "kept_old", "截断影响域外，标签未变，保留旧版")
                report["kept_old"] += 1
                report["entries"].append(probe_id)
                continue
            if _same_content(previous, fresh):
                self._log(previous, "unchanged",
                          "影响域内但距离与见证均未变，保留旧版")
                report["unchanged"] += 1
                report["entries"].append(probe_id)
                continue
            self._append(fresh, previous.ticket)
            if fresh.eligible:
                action = "reissued"
                report["reissued"] += 1
            else:
                action = "lost"
                report["lost"] += 1
            self._log(fresh, action,
                      fresh.reason if not fresh.eligible else "距离或见证变化，重新签发",
                      previous.ticket)
            report["entries"].append(probe_id)
        return report

    @staticmethod
    def _influence_boxes(old_field: SignedDistanceField,
                         new_field: SignedDistanceField) -> list[Box]:
        """标签差异经半径晕圈扩张后的影响域；只有落在其中的标签才重发。"""
        changed = old_field.labels.changed_boxes(new_field.labels)
        if not changed:
            return []
        hr, hc = new_field.halo
        rows, cols = new_field.frame.rows, new_field.frame.cols
        boxes = [
            Box(max(0, piece.row_start - hr - 1), min(rows, piece.row_stop + hr + 1),
                max(0, piece.col_start - hc - 1), min(cols, piece.col_stop + hc + 1))
            for piece in changed
        ]
        # 只合并相交的影响域；相距较远的片段保持独立，避免把整个幅面当成影响域。
        merged: list[Box] = []
        for piece in boxes:
            for index, existing in enumerate(merged):
                if existing.intersected(piece) is not None:
                    merged[index] = existing.union(piece)
                    break
            else:
                merged.append(piece)
        stable = True
        while stable:
            stable = False
            result: list[Box] = []
            for piece in merged:
                for index, existing in enumerate(result):
                    if existing.intersected(piece) is not None:
                        result[index] = existing.union(piece)
                        stable = True
                        break
                else:
                    result.append(piece)
            merged = result
        return merged


def field_for_credentials(inventory, owners, frame, calibration: Calibration,
                          credentials: Iterable[tuple[str, int]],
                          tile_shape=None, scratch_bytes=None) -> SignedDistanceField:
    """按旧瓦片凭据重建距离场，用于回查已交付的目标。"""
    selection = {str(tile_id): int(version) for tile_id, version in credentials}
    labels = LabelMap(frame, owners, inventory, selection)
    options = {}
    if tile_shape is not None:
        options["tile_shape"] = tile_shape
    if scratch_bytes is not None:
        options["scratch_bytes"] = scratch_bytes
    return SignedDistanceField(labels, calibration, **options)


def gauge_of(record: TrainingRecord) -> Gauge:
    return Gauge(record.field.spacing, record.field.radius)
