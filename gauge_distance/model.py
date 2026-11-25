"""量规几何、窗口、瓦片与标定。

坐标约定（与 README 一致）:

* 格点 ``(r, c)`` 的中心在 ``(r * spacing[0], c * spacing[1])`` 微米。
* 定位点按“先行后列、含左不含右”的中心单元确定所属格：格点 ``(r, c)``
  拥有 ``[(r-0.5)*spacing[0], (r+0.5)*spacing[0])`` ×
  ``[(c-0.5)*spacing[1], (c+0.5)*spacing[1])``，即::

      r = floor((y + spacing[0] / 2) / spacing[0])
      c = floor((x + spacing[1] / 2) / spacing[1])

  恰好落在两格共有边界上时归入边界右（下）侧格。
* 坐标 ID 按先行后列排序，即 ``r * ncols + c``；等距见证取 ID 更小的一方。
* 所有平方距离以整数微米平方比较，避免浮点在等距判断上抖动。

瓦片只在其来源的管辖范围内定义前景。越出管辖范围的重叠读回只是上下文，
可以解释缺口，但不能替拥有者定义前景；未回传格一律未知，不默认成背景。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

FOREGROUND = 1
BACKGROUND = 0
UNKNOWN = -1
LABELS = (UNKNOWN, BACKGROUND, FOREGROUND)


# ---------------------------------------------------------------------------
# 量规与标定
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gauge:
    """量规：两轴间距（微米）与截断半径（微米）。

    两轴间距可以不同；训练只需要边界附近一小段距离，超过半径的结果统一
    截断，因此计算窗口只需覆盖半径内的邻域。
    """

    spacing: tuple[int, int]
    radius: int

    def __post_init__(self):
        sr, sc = (int(v) for v in self.spacing)
        if sr <= 0 or sc <= 0:
            raise ValueError("轴间距必须为正")
        if int(self.radius) < 0:
            raise ValueError("截断半径不能为负")
        object.__setattr__(self, "spacing", (sr, sc))
        object.__setattr__(self, "radius", int(self.radius))

    @property
    def radius_squared(self) -> int:
        return self.radius * self.radius

    @property
    def row_spacing(self) -> int:
        return self.spacing[0]

    @property
    def col_spacing(self) -> int:
        return self.spacing[1]

    def center(self, row: int, col: int) -> tuple[int, int]:
        """格点中心坐标 ``(y, x)``，单位微米。"""
        return (row * self.row_spacing, col * self.col_spacing)

    def halo_cells(self) -> tuple[int, int]:
        """半径覆盖的行、列格数（含格点自身）。"""
        return (self.radius // self.row_spacing, self.radius // self.col_spacing)

    def cell_of_point(self, y: float, x: float) -> tuple[int, int]:
        """定位点所属格：按含左不含右的中心单元判定。

        每个格点在格点中心 ``c * spacing`` 周围拥有 ``[c*s - s/2, c*s + s/2)``
        的中心单元；定位点落在共有边界上时归入边界右（下）侧格。定位点与格点
        中心之间允许整数微米偏移，整数输入用精确整数运算避免浮点边界抖动。
        """
        return (self._cell_axis(y, self.row_spacing),
                self._cell_axis(x, self.col_spacing))

    @staticmethod
    def _cell_axis(value: float, spacing: int) -> int:
        """``floor((value + spacing/2) / spacing)``，向左取整。"""
        doubled = float(value) * 2 + spacing
        return int(math.floor(doubled / (2 * spacing)))

    def same_geometry(self, other: "Gauge") -> bool:
        return self.spacing == other.spacing and self.radius == other.radius


@dataclass(frozen=True)
class Calibration:
    """一条测量记录：版本号及其量规。"""

    version: int
    gauge: Gauge

    @property
    def spacing(self) -> tuple[int, int]:
        return self.gauge.spacing

    @property
    def radius(self) -> int:
        return self.gauge.radius

    @classmethod
    def from_record(cls, version: int, spacing: Iterable[int], radius: int) -> "Calibration":
        return cls(int(version), Gauge(tuple(int(v) for v in spacing), int(radius)))


# ---------------------------------------------------------------------------
# 窗口与格点
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coord:
    """完整幅面上的格点坐标与稳定身份。"""

    row: int
    col: int
    identity: int

    @classmethod
    def of(cls, row: int, col: int, ncols: int) -> "Coord":
        return cls(row, col, row * ncols + col)


@dataclass(frozen=True)
class Box:
    """完整幅面内的左闭右开窗口：``[row_start, row_stop) × [col_start, col_stop)``。"""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def __post_init__(self):
        if self.row_stop <= self.row_start or self.col_stop <= self.col_start:
            raise ValueError("窗口必须非空")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_stop - self.row_start, self.col_stop - self.col_start)

    @property
    def area(self) -> int:
        rows, cols = self.shape
        return rows * cols

    def contains(self, row: int, col: int) -> bool:
        return (
            self.row_start <= row < self.row_stop
            and self.col_start <= col < self.col_stop
        )

    def expanded(self, rows: int, cols: int, frame_rows: int, frame_cols: int) -> "Box":
        """向四周扩张并裁剪到完整幅面。"""
        return Box(
            max(0, self.row_start - rows),
            min(frame_rows, self.row_stop + rows),
            max(0, self.col_start - cols),
            min(frame_cols, self.col_stop + cols),
        )

    def intersected(self, other: "Box") -> "Box | None":
        row_start = max(self.row_start, other.row_start)
        row_stop = min(self.row_stop, other.row_stop)
        col_start = max(self.col_start, other.col_start)
        col_stop = min(self.col_stop, other.col_stop)
        if row_stop <= row_start or col_stop <= col_start:
            return None
        return Box(row_start, row_stop, col_start, col_stop)

    def union(self, other: "Box") -> "Box":
        return Box(
            min(self.row_start, other.row_start),
            max(self.row_stop, other.row_stop),
            min(self.col_start, other.col_start),
            max(self.col_stop, other.col_stop),
        )

    def cells(self) -> Iterator[tuple[int, int]]:
        for row in range(self.row_start, self.row_stop):
            for col in range(self.col_start, self.col_stop):
                yield (row, col)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return self.cells()

    def __len__(self) -> int:
        return self.area

    @classmethod
    def of(cls, row: int, col: int) -> "Box":
        return cls(row, row + 1, col, col + 1)

    @classmethod
    def around(cls, row: int, col: int, rows: int, cols: int) -> "Box":
        """以格点为中心、两侧各扩张给定格数的窗口。"""
        return cls(row - rows, row + rows + 1, col - cols, col + cols + 1)

    @classmethod
    def from_record(cls, values: Iterable[int]) -> "Box":
        r0, r1, c0, c1 = (int(v) for v in values)
        return cls(r0, r1, c0, c1)

    def as_record(self) -> list[int]:
        return [self.row_start, self.row_stop, self.col_start, self.col_stop]


@dataclass(frozen=True)
class OwnerRegion:
    """某个瓦片来源在完整幅面上的管辖范围（左闭右开）。"""

    source: str
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def __post_init__(self):
        if self.row_stop <= self.row_start or self.col_stop <= self.col_start:
            raise ValueError(f"来源 {self.source} 的管辖范围为空")

    def contains(self, row: int, col: int) -> bool:
        return (
            self.row_start <= row < self.row_stop
            and self.col_start <= col < self.col_stop
        )

    @property
    def box(self) -> Box:
        return Box(self.row_start, self.row_stop, self.col_start, self.col_stop)

    def overlaps(self, other: "OwnerRegion") -> bool:
        return self.box.intersected(other.box) is not None

    @classmethod
    def from_record(cls, source: str, values: Iterable[int]) -> "OwnerRegion":
        r0, r1, c0, c1 = (int(v) for v in values)
        return cls(str(source), r0, r1, c0, c1)


# ---------------------------------------------------------------------------
# 瓦片
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """一块采集瓦片。

    ``values`` 只覆盖 ``box``。若 ``box`` 越出来源的管辖范围，越界部分只是
    重叠上下文：它可以解释缺口，但不能替拥有者定义前景。
    """

    id: str
    source: str
    version: int
    box: Box
    values: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        rows, cols = self.box.shape
        if len(self.values) != rows or any(len(line) != cols for line in self.values):
            raise ValueError(f"瓦片 {self.id} 版本 {self.version} 的形状与窗口不符")
        for line in self.values:
            for value in line:
                if value not in LABELS:
                    raise ValueError(f"瓦片 {self.id} 含非法标志 {value!r}")

    def label_at(self, row: int, col: int) -> int:
        if not self.box.contains(row, col):
            raise KeyError((row, col))
        return int(self.values[row - self.box.row_start][col - self.box.col_start])

    @property
    def credential(self) -> tuple[str, int]:
        return (self.id, self.version)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.int8)

    @classmethod
    def from_record(cls, record: dict, values) -> "Tile":
        return cls(
            id=str(record["id"]),
            source=str(record["source"]),
            version=int(record["version"]),
            box=Box.from_record(record["box"]),
            values=tuple(tuple(int(v) for v in line) for line in values),
        )


@dataclass(frozen=True)
class TileRecord:
    """交接记录里的一条瓦片条目。"""

    id: str
    source: str
    version: int
    box: Box
    tensor: str

    @classmethod
    def from_record(cls, record: dict) -> "TileRecord":
        return cls(
            id=str(record["id"]),
            source=str(record["source"]),
            version=int(record["version"]),
            box=Box.from_record(record["box"]),
            tensor=str(record["tensor"]),
        )


@dataclass
class TileInventory:
    """按 ``id``、版本归档的瓦片集合。

    同一 ``id`` 的多个版本并存：旧版本仍可按旧凭据回查，新版本只由拥有者
    签发。默认选择每个 ``id`` 的最新版本。
    """

    tiles: dict[str, dict[int, Tile]] = field(default_factory=dict)

    def add(self, tile: Tile) -> None:
        versions = self.tiles.setdefault(tile.id, {})
        existing = versions.get(tile.version)
        if existing is not None and existing != tile:
            raise ValueError(f"瓦片 {tile.id} 版本 {tile.version} 内容冲突")
        versions[tile.version] = tile

    def latest(self, tile_id: str) -> Tile:
        versions = self.tiles[tile_id]
        return versions[max(versions)]

    def get(self, tile_id: str, version: int) -> Tile:
        return self.tiles[tile_id][int(version)]

    def versions(self, tile_id: str) -> list[int]:
        return sorted(self.tiles[tile_id])

    def ids(self) -> list[str]:
        return sorted(self.tiles)

    def selection(self, versions: dict[str, int] | None = None) -> dict[str, int]:
        """解析每个 ``id`` 采用的版本，默认取最新。"""
        chosen = {}
        for tile_id in self.ids():
            if versions is None or tile_id not in versions:
                chosen[tile_id] = max(self.tiles[tile_id])
            else:
                version = int(versions[tile_id])
                if version not in self.tiles[tile_id]:
                    raise KeyError(f"瓦片 {tile_id} 没有版本 {version}")
                chosen[tile_id] = version
        return chosen

    def selected_tiles(self, versions: dict[str, int] | None = None) -> list[Tile]:
        chosen = self.selection(versions)
        return [self.get(tile_id, version) for tile_id, version in sorted(chosen.items())]


# ---------------------------------------------------------------------------
# 完整幅面与权威标志
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """完整采集幅面的格点数。"""

    rows: int
    cols: int

    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("幅面必须非空")

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def box(self) -> Box:
        return Box(0, self.rows, 0, self.cols)

    def identity(self, row: int, col: int) -> int:
        return row * self.cols + col

    def coord(self, row: int, col: int) -> Coord:
        return Coord.of(row, col, self.cols)

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def clip(self, box: Box) -> Box | None:
        return self.box.intersected(box)

    @classmethod
    def from_record(cls, values: Iterable[int]) -> "Frame":
        rows, cols = (int(v) for v in values)
        return cls(rows, cols)


class LabelMap:
    """权威标志查询：只由拥有者定义，管辖范围外的重叠读回不参与。

    未回传、未覆盖的格一律为未知 (``-1``)，从不默认成背景。
    """

    def __init__(
        self,
        frame: Frame,
        owners: Iterable[OwnerRegion],
        inventory: TileInventory,
        selection: dict[str, int] | None = None,
    ):
        self.frame = frame
        self.owners = list(owners)
        self.inventory = inventory
        self.tiles = inventory.selected_tiles(selection)
        self.selection = dict(sorted((tile.id, tile.version) for tile in self.tiles))
        self._by_source: dict[str, list[Tile]] = {}
        for tile in self.tiles:
            self._by_source.setdefault(tile.source, []).append(tile)
        self._arrays = {tile.credential: tile.as_array() for tile in self.tiles}
        self._validate()

    def _validate(self) -> None:
        seen: list[OwnerRegion] = []
        for region in self.owners:
            if self.frame.box.intersected(region.box) is None:
                raise ValueError(f"来源 {region.source} 的管辖范围完全在幅面外")
            for other in seen:
                if region.overlaps(other):
                    raise ValueError(
                        f"来源 {region.source} 与 {other.source} 的管辖范围重叠"
                    )
            seen.append(region)
        regions = {region.source for region in self.owners}
        for source in {tile.source for tile in self.tiles}:
            if source not in regions:
                raise ValueError(f"来源 {source} 没有登记管辖范围")

    def owner_of(self, row: int, col: int) -> OwnerRegion | None:
        for region in self.owners:
            if region.contains(row, col):
                return region
        return None

    def fill(self, box: Box) -> np.ndarray:
        """按拥有者填充窗口标志，管辖外或未回传格为 ``-1``。"""
        if not (
            self.frame.contains(box.row_start, box.col_start)
            and self.frame.contains(box.row_stop - 1, box.col_stop - 1)
        ):
            raise ValueError("窗口超出完整幅面")
        labels = np.full(box.shape, UNKNOWN, dtype=np.int8)
        for region in self.owners:
            piece = box.intersected(region.box)
            if piece is None:
                continue
            for tile in self._by_source.get(region.source, ()):
                overlap = piece.intersected(tile.box)
                if overlap is None:
                    continue
                source = self._arrays[tile.credential][
                    overlap.row_start - tile.box.row_start: overlap.row_stop - tile.box.row_start,
                    overlap.col_start - tile.box.col_start: overlap.col_stop - tile.box.col_start,
                ]
                target = labels[
                    overlap.row_start - box.row_start: overlap.row_stop - box.row_start,
                    overlap.col_start - box.col_start: overlap.col_stop - box.col_start,
                ]
                known_target = target != UNKNOWN
                known_source = source != UNKNOWN
                conflict = known_target & known_source & (target != source)
                if conflict.any():
                    index = np.argwhere(conflict)[0]
                    row = int(overlap.row_start + index[0])
                    col = int(overlap.col_start + index[1])
                    raise ValueError(
                        f"来源 {region.source} 在格 {(row, col)} 给出冲突标志"
                    )
                take = ~known_target & known_source
                target[take] = source[take]
        return labels

    def label_at(self, row: int, col: int) -> int:
        if not self.frame.contains(row, col):
            raise IndexError((row, col))
        return int(self.fill(Box.of(row, col))[0, 0])

    # -- 上下文解释 -------------------------------------------------------

    def coverage(self, row: int, col: int) -> list[dict]:
        """列出所有在该格有读回（含越界的重叠上下文）的瓦片。"""
        holders = []
        for tile in self.tiles:
            if not tile.box.contains(row, col):
                continue
            owner = self.owner_of(row, col)
            pivotal = owner is not None and owner.source == tile.source
            holders.append({
                "tile": tile.id,
                "version": tile.version,
                "source": tile.source,
                "value": tile.label_at(row, col),
                "owner": pivotal,
                "role": "owner" if pivotal else "context",
            })
        return holders

    def explain_cell(self, row: int, col: int) -> dict:
        """解释某格标志的来源：权威值、缺口是否被重叠上下文解释。"""
        if not self.frame.contains(row, col):
            raise IndexError((row, col))
        owner = self.owner_of(row, col)
        holders = self.coverage(row, col)
        owned = [item for item in holders if item["owner"]]
        context = [item for item in holders if not item["owner"]]
        label = self.label_at(row, col)
        defined = bool(owned) and all(item["value"] != UNKNOWN for item in owned)
        context_only = label == UNKNOWN and bool(context)
        if owner is None:
            reason = "该格不在任何来源的管辖范围内，标志为未知"
        elif label == UNKNOWN and not owned:
            reason = "拥有者没有回传该格，标志为未知"
        elif label == UNKNOWN and owned and all(item["value"] == UNKNOWN for item in owned):
            if context:
                reason = "拥有者回传未知；重叠上下文只能解释缺口，不能定义前景"
            else:
                reason = "拥有者回传未知，标志为未知"
        elif label == UNKNOWN:
            reason = "拥有者没有给出该格的确定标志，标志为未知"
        else:
            reason = "拥有者在管辖范围内给出确定标志"
        return {
            "row": row,
            "col": col,
            "owner": None if owner is None else owner.source,
            "label": label,
            "defined_by_owner": defined,
            "context_only": context_only,
            "holders": holders,
            "reason": reason,
        }

    def gap_report(self) -> dict:
        """统计未知格中由重叠上下文覆盖的比例（缺口解释，不是定义）。"""
        total_unknown = 0
        explained = 0
        for row in range(self.frame.rows):
            for col in range(self.frame.cols):
                if self.label_at(row, col) != UNKNOWN:
                    continue
                total_unknown += 1
                if any(not item["owner"] and item["value"] != UNKNOWN
                       for item in self.coverage(row, col)):
                    explained += 1
        return {
            "unknown_cells": total_unknown,
            "explained_by_context": explained,
            "unexplained": total_unknown - explained,
        }

    @property
    def credentials(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.selection.items()))

    def changed_boxes(self, other: "LabelMap", box: Box | None = None,
                      chunk: tuple[int, int] = (256, 256)) -> list[Box]:
        """两个标志图在 ``box`` 内出现差异的紧致行片段。"""
        if self.frame != other.frame:
            raise ValueError("只能比较同一幅面的标志图")
        box = self.frame.box if box is None else box
        clipped = self.frame.clip(box)
        if clipped is None:
            return []
        rows, cols = chunk
        boxes: list[Box] = []
        for row in range(clipped.row_start, clipped.row_stop, rows):
            for col in range(clipped.col_start, clipped.col_stop, cols):
                piece = Box(row, min(row + rows, clipped.row_stop),
                            col, min(col + cols, clipped.col_stop))
                left = self.fill(piece)
                right = other.fill(piece)
                diff = left != right
                if not diff.any():
                    continue
                for dr in range(diff.shape[0]):
                    columns = np.flatnonzero(diff[dr])
                    if columns.size == 0:
                        continue
                    start = int(columns[0])
                    previous = start
                    for value in columns[1:]:
                        value = int(value)
                        if value != previous + 1:
                            boxes.append(Box(piece.row_start + dr, piece.row_start + dr + 1,
                                             piece.col_start + start, piece.col_start + previous + 1))
                            start = value
                        previous = value
                    boxes.append(Box(piece.row_start + dr, piece.row_start + dr + 1,
                                     piece.col_start + start, piece.col_start + previous + 1))
        return boxes

    def changed_count(self, other: "LabelMap", box: Box | None = None,
                      chunk: tuple[int, int] = (256, 256)) -> int:
        total = 0
        for piece in self.changed_boxes(other, box, chunk):
            total += piece.area
        return total
