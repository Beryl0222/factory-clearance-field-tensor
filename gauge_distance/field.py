"""截断带符号欧氏距离场。

语义:

* 已知前景格的距离为正，已知背景格的距离为负；距离取到最近已知异类格的
  格点中心。平方距离用整数微米平方比较，等距取坐标 ID（先行后列）更小者。
* 距离严格超过截断半径的候选不在截断域内。半径内没有已知异类格、也没有
  未知格时，结果为带符号阈值（截断无见证）。恰等于阈值的候选仍在域内。
* 半径内存在未知格，且该格取异类会得到更小距离、或等距但 ID 更小的见证、
  或本来没有见证而它能提供见证时，结果标为无法确定。
* 未知查询格不产生距离目标。
* 计算只在请求窗口及其半径晕圈内逐块进行，临时内存受工作区预算约束，不按
  完整幅面分配“格点 × 边界候选”的组合。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

import numpy as np

from .model import (
    BACKGROUND,
    FOREGROUND,
    UNKNOWN,
    Box,
    Calibration,
    Coord,
    Gauge,
    LabelMap,
)

INF = np.iinfo(np.int64).max // 8
DEFAULT_SCRATCH_BYTES = 4 * 1024 * 1024
DEFAULT_TILE_SHAPE = (96, 64)

#: 每个展开格在最近点变换与状态判定中保留的 int64 面板数（实测约 133 字节/格，
#: 这里留余量，保证单块峰值不越过工作区预算）。
_PANELS = 160


class FieldStatus(IntEnum):
    NO_TARGET = 0
    TRUSTED = 1
    TRUNCATED_NO_WITNESS = 2
    UNDETERMINED = 3

    TRUNCATED = 2
    UNKNOWN = 3


class Uncertainty(IntEnum):
    NONE = 0
    UNKNOWN_COULD_BE_CLOSER = 1
    UNKNOWN_WITNESS_TIE = 2
    UNKNOWN_COULD_SUPPLY_WITNESS = 3


STATUS_REASONS = {
    FieldStatus.NO_TARGET: "查询格标志未知，不产生距离目标",
    FieldStatus.TRUSTED: "最近的已知异类格给出确定距离与见证",
    FieldStatus.TRUNCATED_NO_WITNESS: "截断半径内没有已知异类格，也没有未知格",
    FieldStatus.UNDETERMINED: "截断半径内的未知格可能改变距离或见证",
}

UNCERTAINTY_REASONS = {
    Uncertainty.NONE: "",
    Uncertainty.UNKNOWN_COULD_BE_CLOSER: "半径内未知格若取异类会得到更小距离",
    Uncertainty.UNKNOWN_WITNESS_TIE: "半径内未知格等距且坐标 ID 更小，见证不唯一",
    Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS: "半径内未知格可能提供见证，无法断言无见证",
}


@dataclass(frozen=True)
class CellResult:
    row: int
    col: int
    identity: int
    label: int
    sign: int
    status: FieldStatus
    squared: int | float | None
    value: float | None
    witness: Coord | None
    uncertainty: Uncertainty = Uncertainty.NONE
    reason: str = ""

    @property
    def definite(self) -> bool:
        return self.status in (FieldStatus.TRUSTED, FieldStatus.TRUNCATED_NO_WITNESS)

    @property
    def has_witness(self) -> bool:
        return self.status is FieldStatus.TRUSTED and self.witness is not None

    @property
    def eligible_for_training(self) -> bool:
        """训练只接受可证实的符号距及坐标见证。"""
        return self.has_witness and self.value is not None


def reason_for(status: FieldStatus, uncertainty: Uncertainty = Uncertainty.NONE) -> str:
    base = STATUS_REASONS[status]
    if uncertainty is Uncertainty.NONE:
        return base
    return f"{base}：{UNCERTAINTY_REASONS[uncertainty]}"


def _cell(row, col, identity, label, sign, status, squared, value, witness,
          uncertainty=Uncertainty.NONE) -> CellResult:
    return CellResult(row, col, identity, label, sign, status, squared, value,
                      witness, uncertainty, reason_for(status, uncertainty))


# ---------------------------------------------------------------------------
# 精确最近点变换（逐轴分离，整数平方距离）
# ---------------------------------------------------------------------------


def _nearest_in_rows(mask: np.ndarray, spacing: int):
    """同行内最近格的平方水平距离与列见证；等距取更小列。"""
    _, cols = mask.shape
    index = np.arange(cols, dtype=np.int64)
    left = np.where(mask, index, -1)
    np.maximum.accumulate(left, axis=1, out=left)
    right = np.where(mask, index, cols)
    right = np.minimum.accumulate(right[:, ::-1], axis=1)[:, ::-1]
    has_left = left >= 0
    has_right = right < cols
    step = np.int64(spacing)
    left_delta = index - np.where(has_left, left, 0)
    right_delta = np.where(has_right, right, 0) - index
    left_sq = np.where(has_left, (left_delta * step) ** 2, INF)
    right_sq = np.where(has_right, (right_delta * step) ** 2, INF)
    take_left = left_sq <= right_sq
    squared = np.where(take_left, left_sq, right_sq)
    witness = np.where(take_left, left, right)
    none = ~(has_left | has_right)
    squared = np.where(none, INF, squared).astype(np.int64, copy=False)
    witness = np.where(none, -1, witness).astype(np.int64, copy=False)
    return squared, witness


def _nearest_on_rows(horizontal: np.ndarray, witness_col: np.ndarray,
                     spacing: int, halo_rows: int):
    """沿行合并；等距取更小行，因此也是更小坐标 ID。"""
    rows, cols = horizontal.shape
    best = np.full((rows, cols), INF, dtype=np.int64)
    best_row = np.full((rows, cols), -1, dtype=np.int64)
    best_col = np.full((rows, cols), -1, dtype=np.int64)
    step = np.int64(spacing)
    for offset in range(-halo_rows, halo_rows + 1):
        if offset >= 0:
            dest = slice(0, rows - offset if offset else rows)
            source = slice(offset, rows)
        else:
            dest = slice(-offset, rows)
            source = slice(0, rows + offset)
        if source.stop <= source.start:
            continue
        cost = np.int64(offset * step)
        cost = cost * cost
        candidate = horizontal[source] + cost
        np.minimum(candidate, INF, out=candidate)
        old = best[dest]
        src_rows = np.arange(source.start, source.stop, dtype=np.int64)[:, None]
        src_cols = witness_col[source]
        take = candidate < old
        tie = (candidate == old) & (src_rows < best_row[dest])
        take |= tie
        best[dest] = np.where(take, candidate, old)
        best_row[dest] = np.where(take, src_rows, best_row[dest])
        best_col[dest] = np.where(take, src_cols, best_col[dest])
    valid = best < INF
    return best, np.where(valid, best_row, -1), np.where(valid, best_col, -1)


def _nearest_sites(mask: np.ndarray, spacing: tuple[int, int], halo_rows: int):
    horizontal, witness_col = _nearest_in_rows(mask, spacing[1])
    return _nearest_on_rows(horizontal, witness_col, spacing[0], halo_rows)


# ---------------------------------------------------------------------------
# 距离场
# ---------------------------------------------------------------------------


class SignedDistanceField:
    """基于权威标志图与量规的截断带符号距离场。"""

    def __init__(self, labels: LabelMap, calibration: Calibration, *,
                 tile_shape: tuple[int, int] | None = None,
                 scratch_bytes: int = DEFAULT_SCRATCH_BYTES):
        self.labels = labels
        self.calibration = calibration
        self.gauge: Gauge = calibration.gauge
        self.scratch_bytes = int(scratch_bytes)
        if tile_shape is None:
            tile_shape = self._fit_tile_shape()
        self.tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        if self.tile_shape[0] <= 0 or self.tile_shape[1] <= 0:
            raise ValueError("块形状必须为正")
        self._check_scratch()

    @property
    def frame(self):
        return self.labels.frame

    @property
    def credentials(self) -> tuple[tuple[str, int], ...]:
        return self.labels.credentials

    @property
    def halo(self) -> tuple[int, int]:
        return self.gauge.halo_cells()

    def _check_scratch(self) -> None:
        rows, cols = self.tile_shape
        hr, hc = self.halo
        expanded = (rows + 2 * hr) * (cols + 2 * hc)
        estimate = expanded * _PANELS
        if estimate > self.scratch_bytes:
            raise ValueError(
                f"单块临时内存估计 {estimate} 字节超过工作区预算 {self.scratch_bytes}；"
                "请减小 tile_shape 或提高预算"
            )

    def _fit_tile_shape(self) -> tuple[int, int]:
        """按工作区预算选取最大可用块形状；放不下半径晕圈则拒绝。"""
        hr, hc = self.halo
        rows, cols = DEFAULT_TILE_SHAPE
        budget_cells = self.scratch_bytes // _PANELS
        while True:
            if (rows + 2 * hr) * (cols + 2 * hc) <= budget_cells:
                return (rows, cols)
            if rows <= 1 and cols <= 1:
                break
            if rows >= cols and rows > 1:
                rows //= 2
            elif cols > 1:
                cols //= 2
            else:
                rows //= 2
        raise ValueError(
            f"工作区预算 {self.scratch_bytes} 字节放不下半径 {self.gauge.radius} "
            "所需的晕圈；无法在限定内存内计算距离场"
        )

    # -- 输出数组路径 -----------------------------------------------------

    def _tile_payload(self, tile: Box) -> dict[str, np.ndarray]:
        """计算一块窗口的结果数组（只分配窗口加晕圈大小）。"""
        expanded = tile.expanded(*self.halo, self.frame.rows, self.frame.cols)
        labels = self.labels.fill(expanded)
        gauge = self.gauge
        hr, _ = self.halo
        fg = labels == FOREGROUND
        bg = labels == BACKGROUND
        unknown = labels == UNKNOWN
        to_bg_sq, to_bg_row, to_bg_col = _nearest_sites(bg, gauge.spacing, hr)
        to_fg_sq, to_fg_row, to_fg_col = _nearest_sites(fg, gauge.spacing, hr)
        un_sq, un_row, un_col = _nearest_sites(unknown, gauge.spacing, hr)

        rs = tile.row_start - expanded.row_start
        cs = tile.col_start - expanded.col_start
        re = rs + tile.shape[0]
        ce = cs + tile.shape[1]
        q = labels[rs:re, cs:ce]
        query_fg = q == FOREGROUND
        known = q != UNKNOWN

        known_sq = np.where(query_fg, to_bg_sq[rs:re, cs:ce], to_fg_sq[rs:re, cs:ce])
        known_row = np.where(query_fg, to_bg_row[rs:re, cs:ce], to_fg_row[rs:re, cs:ce])
        known_col = np.where(query_fg, to_bg_col[rs:re, cs:ce], to_fg_col[rs:re, cs:ce])
        u_sq = un_sq[rs:re, cs:ce]
        u_row = un_row[rs:re, cs:ce]
        u_col = un_col[rs:re, cs:ce]

        ncols = self.frame.cols
        cutoff = gauge.radius_squared
        known_id = known_row * ncols + known_col
        unknown_id = u_row * ncols + u_col
        has_known = known & (known_row >= 0) & (known_sq <= cutoff)
        has_unknown = (u_row >= 0) & (u_sq <= cutoff)
        closer = has_known & has_unknown & (u_sq < known_sq)
        tie = has_known & has_unknown & (u_sq == known_sq) & (unknown_id < known_id)
        supply = (~has_known) & has_unknown
        undetermined = closer | tie | supply
        truncated = known & (~has_known) & (~has_unknown)
        trusted = has_known & (~undetermined)
        sign = np.where(query_fg, 1, -1).astype(np.int8)

        status = np.full(q.shape, FieldStatus.NO_TARGET, dtype=np.int8)
        status[known] = FieldStatus.TRUNCATED_NO_WITNESS
        status[truncated] = FieldStatus.TRUNCATED_NO_WITNESS
        status[trusted] = FieldStatus.TRUSTED
        status[undetermined] = FieldStatus.UNDETERMINED
        status[~known] = FieldStatus.NO_TARGET

        uncertainty = np.zeros(q.shape, dtype=np.int8)
        uncertainty[closer] = Uncertainty.UNKNOWN_COULD_BE_CLOSER
        uncertainty[tie] = Uncertainty.UNKNOWN_WITNESS_TIE
        uncertainty[supply] = Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS
        uncertainty[~undetermined] = Uncertainty.NONE

        value = np.zeros(q.shape, dtype=np.float64)
        value[truncated] = gauge.radius * sign[truncated].astype(np.float64)
        value[trusted] = np.sqrt(known_sq[trusted].astype(np.float64)) * sign[trusted]

        squared = np.full(q.shape, -1, dtype=np.int64)
        squared[trusted] = known_sq[trusted]

        # 最近点变换在展开块内以局部坐标运行；行、列的常量原点偏移不改变
        # 坐标 ID 的先后关系，但输出见证时必须换回完整幅面坐标。
        witness_row = np.where(trusted, known_row + expanded.row_start, -1).astype(np.int64)
        witness_col = np.where(trusted, known_col + expanded.col_start, -1).astype(np.int64)
        witness_id = np.where(trusted, witness_row * ncols + witness_col, -1)

        return {
            "box": tile,
            "label": q.astype(np.int8, copy=True),
            "sign": sign,
            "status": status,
            "uncertainty": uncertainty,
            "value": value,
            "squared": squared,
            "witness_row": witness_row,
            "witness_col": witness_col,
            "witness_id": witness_id.astype(np.int64),
        }

    def _tiles(self, box: Box) -> Iterator[Box]:
        rows, cols = self.tile_shape
        for row in range(box.row_start, box.row_stop, rows):
            for col in range(box.col_start, box.col_stop, cols):
                yield Box(row, min(row + rows, box.row_stop),
                          col, min(col + cols, box.col_stop))

    def iter_tiles(self, box: Box) -> Iterator[tuple[Box, dict[str, np.ndarray]]]:
        """逐块流式给出结果数组，内存只与块大小和晕圈有关。

        返回数组的相对坐标已换回完整幅面坐标；``label`` 是该块的权威标志。
        """
        clipped = self.frame.clip(box)
        if clipped is None:
            return
        for tile in self._tiles(clipped):
            payload = self._tile_payload(tile)
            yield tile, payload

    def write(self, box: Box, target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """把窗口结果写入调用方预分配的数组（避免整幅面组合分配）。

        ``target`` 需含 ``status`` (int8)、``value`` (float64)、``squared``
        (int64，无见证为 -1)、``witness_row`` / ``witness_col`` (int64，无见证
        为 -1)、``uncertainty`` (int8)。窗口外的目标区域填无目标值。
        """
        shape = box.shape
        for key, dtype in (("status", np.int8), ("value", np.float64),
                           ("squared", np.int64), ("witness_row", np.int64),
                           ("witness_col", np.int64), ("uncertainty", np.int8)):
            if target[key].shape != shape:
                raise ValueError(f"目标数组 {key} 形状应为 {shape}")
        target["status"].fill(FieldStatus.NO_TARGET)
        target["value"].fill(0.0)
        target["squared"].fill(-1)
        target["witness_row"].fill(-1)
        target["witness_col"].fill(-1)
        target["uncertainty"].fill(Uncertainty.NONE)
        clipped = self.frame.clip(box)
        if clipped is None:
            return target
        for tile in self._tiles(clipped):
            payload = self._tile_payload(tile)
            r0 = tile.row_start - box.row_start
            c0 = tile.col_start - box.col_start
            rows, cols = tile.shape
            for key in ("status", "value", "squared", "witness_row",
                        "witness_col", "uncertainty"):
                target[key][r0:r0 + rows, c0:c0 + cols] = payload[key]
        return target

    def arrays(self, box: Box) -> dict[str, np.ndarray]:
        """分配并返回窗口结果数组；形状仅为请求窗口。"""
        return self.write(box, {
            "status": np.empty(box.shape, np.int8),
            "value": np.empty(box.shape, np.float64),
            "squared": np.empty(box.shape, np.int64),
            "witness_row": np.empty(box.shape, np.int64),
            "witness_col": np.empty(box.shape, np.int64),
            "uncertainty": np.empty(box.shape, np.int8),
        })

    # -- 逐格对象路径 -----------------------------------------------------

    def cells(self, box: Box) -> Iterator[CellResult]:
        clipped = self.frame.clip(box)
        if clipped is None:
            return
        ncols = self.frame.cols
        for tile in self._tiles(clipped):
            payload = self._tile_payload(tile)
            for dr in range(tile.shape[0]):
                row = tile.row_start + dr
                for dc in range(tile.shape[1]):
                    col = tile.col_start + dc
                    status = FieldStatus(int(payload["status"][dr, dc]))
                    uncertainty = Uncertainty(int(payload["uncertainty"][dr, dc]))
                    squared = int(payload["squared"][dr, dc])
                    wr = int(payload["witness_row"][dr, dc])
                    wc = int(payload["witness_col"][dr, dc])
                    yield _cell(
                        row, col, row * ncols + col,
                        int(payload["label"][dr, dc]),
                        int(payload["sign"][dr, dc]),
                        status,
                        squared if squared >= 0 else None,
                        float(payload["value"][dr, dc]) if status in
                        (FieldStatus.TRUSTED, FieldStatus.TRUNCATED_NO_WITNESS) else None,
                        Coord(wr, wc, wr * ncols + wc) if status is FieldStatus.TRUSTED else None,
                        uncertainty,
                    )

    def region(self, box: Box) -> list[CellResult]:
        return list(self.cells(box))

    def features(self, box: Box) -> dict[str, np.ndarray]:
        """窗口的数组结果：状态、可信值、平方距离、见证 ID 与待判定原因。

        数组只覆盖请求窗口；``witness_id`` 无见证为 -1。
        """
        out = self.arrays(box)
        ncols = self.frame.cols
        out["witness_id"] = np.where(
            out["witness_row"] >= 0,
            out["witness_row"] * ncols + out["witness_col"],
            -1,
        )
        out["cell_id"] = (
            np.arange(box.row_start, box.row_stop, dtype=np.int64)[:, None] * ncols
            + np.arange(box.col_start, box.col_stop, dtype=np.int64)[None, :]
        )
        return out

    def summarize(self, box: Box) -> dict:
        """窗口内各状态的格数与待判定原因计数。"""
        out = self.features(box)
        status = out["status"]
        counts = {member.name.lower(): int(np.count_nonzero(status == member))
                  for member in FieldStatus}
        uncertainty = {
            member.name.lower(): int(np.count_nonzero(out["uncertainty"] == member))
            for member in Uncertainty
            if member is not Uncertainty.NONE
        }
        return {
            "cells": int(status.size),
            "status": counts,
            "uncertainty": uncertainty,
        }

    def cell(self, row: int, col: int) -> CellResult:
        if not self.frame.contains(row, col):
            raise IndexError((row, col))
        return next(iter(self.cells(Box.of(row, col))))

    # -- 定位点 -----------------------------------------------------------

    def probe(self, y: float, x: float) -> CellResult:
        """定位点的距离目标：所在格定符号，距离与见证用实际定位坐标。

        定位点与格点中心之间允许整数微米偏移；整数坐标用整数平方距离比较，
        非整数坐标用浮点平方距离（仍按相等才视为等距）。格点中心的见证只在
        本定位点上使用，不整格共用。
        """
        gauge = self.gauge
        row, col = gauge.cell_of_point(y, x)
        if not self.frame.contains(row, col):
            raise IndexError((row, col))
        label = self.labels.label_at(row, col)
        identity = row * self.frame.cols + col
        if label == UNKNOWN:
            return _cell(row, col, identity, label, 0,
                         FieldStatus.NO_TARGET, None, None, None)
        opposite = BACKGROUND if label == FOREGROUND else FOREGROUND
        sign = 1 if label == FOREGROUND else -1
        hr, hc = self.halo
        box = self.frame.clip(Box(row - hr - 1, row + hr + 2,
                                  col - hc - 1, col + hc + 2))
        if box is None:
            raise IndexError((row, col))
        labels = self.labels.fill(box)
        ncols = self.frame.cols
        cutoff = gauge.radius_squared

        exact = float(y).is_integer() and float(x).is_integer()
        if exact:
            y_int, x_int = int(y), int(x)

        def squared_to(r: int, c: int):
            cy, cx = gauge.center(r, c)
            if exact:
                dy = y_int - cy
                dx = x_int - cx
                return dy * dy + dx * dx
            dy = y - cy
            dx = x - cx
            return dy * dy + dx * dx

        def scan(kind: int):
            best = None
            for r in range(box.row_start, box.row_stop):
                for c in range(box.col_start, box.col_stop):
                    if int(labels[r - box.row_start, c - box.col_start]) != kind:
                        continue
                    value = squared_to(r, c)
                    ident = r * ncols + c
                    if best is None or value < best[0] or (value == best[0] and ident < best[1]):
                        best = (value, ident, Coord(r, c, ident))
            return best

        known = scan(opposite)
        unknown = scan(UNKNOWN)
        known_in = known is not None and known[0] <= cutoff
        unknown_in = unknown is not None and unknown[0] <= cutoff
        if known_in and unknown_in and unknown[0] < known[0]:
            return _cell(row, col, identity, label, sign, FieldStatus.UNDETERMINED,
                         None, None, None, Uncertainty.UNKNOWN_COULD_BE_CLOSER)
        if known_in and unknown_in and unknown[0] == known[0] and unknown[1] < known[1]:
            return _cell(row, col, identity, label, sign, FieldStatus.UNDETERMINED,
                         None, None, None, Uncertainty.UNKNOWN_WITNESS_TIE)
        if known_in:
            sq = known[0]
            squared = int(sq) if float(sq).is_integer() else float(sq)
            return _cell(row, col, identity, label, sign, FieldStatus.TRUSTED,
                         squared, math.sqrt(sq) * sign, known[2])
        if unknown_in:
            return _cell(row, col, identity, label, sign, FieldStatus.UNDETERMINED,
                         None, None, None, Uncertainty.UNKNOWN_COULD_SUPPLY_WITNESS)
        return _cell(row, col, identity, label, sign,
                     FieldStatus.TRUNCATED_NO_WITNESS, None,
                     float(gauge.radius) * sign, None)
