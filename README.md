# 量规修订后的截断带符号距离变换

小幅合成缺陷图按读头权属给出前景标志和未知位置。距离量规使用微米。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

位图编码 1 为前景、0 为背景、-1 为未知。已知格才产生距离目标，前景距离为正，背景距离为负。格点中心在整数格位置乘各轴间距。定位申请的坐标以行向和列向微米给出，以含左边不含右边的中心单元确定所属格，偏移不会改变原网格或面积。符号距离为平方比较后取平方根，定位点采用其自身坐标。阈值及与阈值恰等的候选均在截断域内。域外无见证的确定结果为带符号阈值。
坐标 ID 按先行后列排序，未知格在成为相反类别时可能改变见证也算不确定。瓦片记录仅由拥有者生效，窗口区间左闭右开。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。

## 距离场接口

`gauge_distance` 包把上述约定做成可调用接口，实现细节与测试位于 `gauge_distance/` 与 `tests/`。

```python
from gauge_distance import load_capture

capture = load_capture("data/capture.json")
field = capture.field()                    # 默认每个 id 的最新瓦片版本
cell = field.cell(2, 3)                    # 格点距离目标
probe = field.probe(4, 7)                   # 定位点距离目标（用实际坐标）
grid = field.features(capture.frame.box)    # 状态 / 值 / 见证数组
```

- **格点距离**：`cell(r, c)`、`region(box)`、`features(box)`、`arrays(box)`、
  `iter_tiles(box)`、`write(box, target)`。距离取到最近已知异类格的格点中心，
  平方距离以整数微米平方比较；等距时取坐标 ID（先行后列）更小者。
- **四种状态**（`FieldStatus`）：
  - `TRUSTED`：有确定见证，记录平方距离与见证坐标，可用于训练。
  - `TRUNCATED_NO_WITNESS`：半径内既无已知异类格也无未知格，值为带符号阈值；
    确定但无见证，不可用作训练目标。
  - `UNDETERMINED`：半径内存在未知格，且它若取异类会带来更小距离、等距但
    ID 更小的见证、或本来没有见证而能提供见证。`uncertainty` 给出具体原因。
  - `NO_TARGET`：查询格本身未知，不产生距离目标。
- **定位点**：`probe(y, x)` 按中心单元确定所属格与符号，距离与见证用定位点
  自身的微米坐标，格点中心的见证不会整格共用。整数坐标用精确整数平方距离。
- **量规**：`Gauge(spacing, radius)`，两轴间距独立，距离严格超过半径才截断。
  `capture.field(calibration=...)` 可切到记录里的新量规；旧瓦片不能与新量规
  拼在同一份距离场里由调用方显式选择。
- **管权与缺口**：瓦片只在其来源的管辖范围内定义标志；越界重叠读回只作上下文，
  `LabelMap.explain_cell` / `gap_report` 说明未回传格是否被上下文覆盖，但值仍为
  未知，从不默认成背景。
- **版本与回查**：`field.credentials` 是采用的瓦片 `(id, version)`；
  `capture.field(selection=...)` 可按旧凭据重建；`field_version(field)` 给出包含
  瓦片内容摘要的版本号。
- **训练记录**：`training.record_probe` / `evaluate_probe` 生成绑定距离场版本的
  记录；`verify_record` 拒绝量规不符、凭据或内容不符、以及不再可证实的记录。
  `assemble_sample` 拒绝新旧量规混拼。`TrainingLedger` 登记签发、按影响域重发、
  保留旧版并说明原因，支持按旧凭据回查（`records_with_credentials`）与追查
  （`explain`）。
- **工作区**：`SignedDistanceField(..., scratch_bytes=..., tile_shape=...)`
  只在请求窗口加半径晕圈内逐块计算。未指定块形状时按预算自动选取；连最小块加
  晕圈都放不下时直接报错，不会退回“整幅面 × 边缘候选”的分配。

`python -m gauge_distance.cli data/capture.json` 打印窗口汇总，`--probes`、
`--prior`、`--gauge-change`、`--reissue-prior` 分别查看定位记录、旧版本、新量规
与重发报告。
