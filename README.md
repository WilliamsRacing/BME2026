# 传感器数据 DataLoader

从华为传感器 ZIP 中按时间窗口读取 ACC、GYRO、PPG，并关联用户资料与用餐标签。输出为 Python 字典和 NumPy 数组，不依赖 PyTorch、TensorFlow 或 GPU。

以传感器数据为主表：没有对应传感器数据的用户资料和用餐记录直接忽略。原 ZIP 始终保留，读取时按需解压解析到临时数组，读取完成后自动清理，无需全量解压或重新压缩。

## 1. 推荐目录结构

```text
BME2026/
├── README.md
├── requirements-dataloader.txt
├── demo_dataloader.py                  # 命令行读取示例
├── dataTables/                        # CSV 数据表
│   ├── 传感器原始数据.csv               # 必需：传感器主表、ZIP 路径与用户标识
│   ├── 用餐信息1.csv                   # 必需：用餐时间区间及事件属性
│   ├── 用户信息.csv                    # 必需：用户资料
│   └── 用餐信息2.csv                   # 可保留；当前加载器不读取、不重复合并
├── t_zsstnnrj_sensororiginaldata_system附件0916_1948/
│   ├── sensorData-....zip              # 原始附件，不需要提前解压
│   └── ...
├── sensor_loader/
│   ├── __init__.py                     # 对外接口
│   ├── catalog.py                      # 数据表读取、关联、分组划分
│   ├── dataset.py                      # ZIP 解析、时间窗口、批处理、清理
│   └── README.md                       # 实现细节与限制
├── tests/
│   └── test_sensor_loader.py
├── docs/
│   └── 数据结构与DataLoader设计.md
├── 样板/                              # 可选：用于理解原始文件结构
└── .sensor_loader_tmp/                 # 运行时临时目录，内容自动清理
```

路径规则：

- `build_catalog(root)` 优先从 `root/dataTables/` 读取表；只有不存在该目录时才从 `root/` 读取。若 `dataTables` 已存在但缺少必需 CSV，会明确报错，不混用其他位置的表。
- ZIP 默认在项目根目录的直接子目录中查找。只有一个子目录含 `sensorData-*.zip` 时自动选择；没有或存在多个候选时，需要指定 `zip_dir`。
- ZIP 应直接放在所选附件目录下，当前不递归搜索其子目录。文件名必须与传感器表 `sensorData` 字段的最后一段一致。
- 可以把附件目录命名为 `sensor_zips/` 等更短的名称，文件名无需更改。本文示例保留当前目录名称。
- Python 接口支持显式 `table_dir` 和 `zip_dir`。建议传入绝对路径；相对路径按进程当前工作目录解释。

## 2. 环境与安装

参考`requirements.txt`

## 3. 如何运行

### 快速检查：一个 ZIP、两个批次

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py
```

默认从 `dataTables` 读取元数据，读取一个非测试 ZIP 的最多两个批次；每批最多 4 个窗口，每个窗口 30 秒，步长 15 秒。终端输出关联统计、三路信号的形状和类型、标签、源文件及临时空间峰值。demo 只打印摘要，完整数组需通过 Python 接口获取。

### 指定数据位置

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py --root 'D:\William\code\BME\BME2026' --zip-dir 'D:\William\code\BME\BME2026\t_zsstnnrj_sensororiginaldata_system附件0916_1948'
```

`--root` 指包含 `dataTables` 的项目目录，而不是 `dataTables` 本身。demo 的默认 root 是脚本所在目录。

### 调整窗口和输出类型

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py --batch-size 8 --window-seconds 30 --stride-seconds 15 --normalize window_zscore
```

默认 `--normalize none` 返回原始 `int64` 信号。`window_zscore` 按每个窗口、每个通道标准化：先以 `float64` 计算均值和标准差，再输出 `float32`。它会移除窗口的绝对基线和幅度，是否启用由后续训练方案决定。

### 顺序读取所有 ZIP

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py --max-files 0 --max-batches 0
```

两个 `0` 分别表示不限制文件数、不限制批次数。读取全部数据需要时间，但不会同时解压全部文件。默认忽略 externalid 为 `test` 的传感器记录；加 `--include-test-records` 可保留。

### 常用命令行参数

| 参数                     | 默认值        | 用途                                              |
| ------------------------ | ------------- | ------------------------------------------------- |
| `--root`                 | demo 所在目录 | 项目/数据根目录                                   |
| `--zip-dir`              | 自动发现      | 原始 ZIP 所在目录                                 |
| `--batch-size`           | `4`           | 每批最多多少个窗口                                |
| `--window-seconds`       | `30`          | 时间窗口长度，秒                                  |
| `--stride-seconds`       | `15`          | 窗口移动步长，秒                                  |
| `--max-files`            | `1`           | 选择多少个 ZIP，`0` 为全部                        |
| `--max-batches`          | `2`           | 输出多少个批次，`0` 为全部                        |
| `--recording-id`         | 不限制        | 指定传感器表 uniqueid，可重复传入                 |
| `--split`                | `all`         | `all`、`train`、`val`、`test`；不是按窗口随机划分 |
| `--normalize`            | `none`        | `none` 或 `window_zscore`                         |
| `--temp-gib`             | `2`           | 每个读取器的临时数组空间上限，GiB                 |
| `--include-test-records` | 不启用        | 保留 externalid 为 test 的传感器记录              |

`--split train` 只改变候选用户；若需遍历该集合的全部文件和批次，还要设置 `--max-files 0 --max-batches 0`。

## 4. 在 Python 中获取数据

以下代码可直接在项目根目录运行。示例只读取一个批次，提前退出也会清理临时文件。

```python
from pathlib import Path
from sensor_loader import build_catalog, LoaderConfig, ZipWindowDataset, SensorDataLoader

root = Path(r"D:\William\code\BME\BME2026")
catalog = build_catalog(
    root,
    table_dir=root / "dataTables",
    zip_dir=root / "t_zsstnnrj_sensororiginaldata_system附件0916_1948",
)

config = LoaderConfig(
    window_seconds=30,
    stride_seconds=15,
    normalization="none",              # 原始 int64；也可用 window_zscore
    shuffle_files=False,
    shuffle_windows=False,
    max_temp_bytes=2 * 1024**3,
)
dataset = ZipWindowDataset(catalog, config)

with SensorDataLoader(
    dataset, batch_size=4, temp_root=root / ".sensor_loader_tmp"
) as loader:
    for batch in loader:
        acc = batch["signals"]["acc"]
        gyro = batch["signals"]["gyro"]
        ppg = batch["signals"]["ppg"]
        labels = batch["target"]["label"]
        label_mask = batch["target"]["label_mask"]

        print(acc.shape, acc.dtype)
        print(gyro.shape, gyro.dtype)
        print(ppg.shape, ppg.dtype)
        print(labels, label_mask)
        break

# batch 中的数组是独立内存副本，退出 with 后仍可使用。
```

必须使用 `with`，确保正常结束、提前 break 或异常时关闭文件并清理。程序被操作系统强杀或断电时可能留下临时目录，不能保证执行清理。

如需按用户组划分，使用 `split_healthids(catalog, seed=42)`，再将相应列表传给 `ZipWindowDataset(..., healthids=splits['train'])`。该功能不自动验证真实受试者身份，正式实验前需核对用户映射。改变 epoch 的打乱顺序可调用 `dataset.set_epoch(epoch)`；打乱需在 LoaderConfig 中开启。

## 5. 输出结构和数据类型

一次迭代返回一个 `dict`，表示一个批次。数值部分全部是 `numpy.ndarray`，不是训练框架张量。

定义：`B` 为本批窗口数，`Tacc/Tgyro/Tppg` 是各模态在本批 padding 后的最大时间点数，`Cppg` 默认为 44。

```text
batch: dict
├── signals: dict
│   ├── acc                  ndarray [B, 3, Tacc]
│   ├── gyro                 ndarray [B, 3, Tgyro]
│   └── ppg                  ndarray [B, Cppg, Tppg]
├── valid_mask: dict         acc / gyro / ppg
├── channel_mask: dict       acc / gyro / ppg
├── timestamp_ms: dict       acc / gyro / ppg
├── relative_time_s: dict    acc / gyro / ppg
├── lengths: dict            acc / gyro / ppg
├── target: dict
│   ├── label
│   ├── label_mask
│   └── meal_fraction
└── meta: list[dict]         B 个窗口的溯源信息
```

### 数值字段

以下 `m` 表示 `acc`、`gyro` 或 `ppg`；`C`、`T` 为该模态的通道数和时间长度。

| 访问方式                           | 数据类型 / dtype     | 形状              | 含义                                         |
| ---------------------------------- | -------------------- | ----------------- | -------------------------------------------- |
| `batch['signals']['acc']`          | ndarray，默认`int64` | `[B, 3, Tacc]`    | ACC_X、ACC_Y、ACC_Z，顺序固定                |
| `batch['signals']['gyro']`         | ndarray，默认`int64` | `[B, 3, Tgyro]`   | GYRO_X、GYRO_Y、GYRO_Z，顺序固定             |
| `batch['signals']['ppg']`          | ndarray，默认`int64` | `[B, Cppg, Tppg]` | 默认 PPG1–PPG44，按配置顺序输出              |
| `batch['valid_mask'][m]`           | ndarray，`bool`      | `[B, C, T]`       | 有效时间点与通道的联合掩码；padding 为 False |
| `batch['channel_mask'][m]`         | ndarray，`bool`      | `[B, C]`          | 当前 TXT 中该通道是否出现非零值              |
| `batch['timestamp_ms'][m]`         | ndarray，`int64`     | `[B, T]`          | 原始 Unix 毫秒批次时间戳，padding 为 0       |
| `batch['relative_time_s'][m]`      | ndarray，`float32`   | `[B, T]`          | 相对窗口起点的秒数，padding 为 0             |
| `batch['lengths'][m]`              | ndarray，`int64`     | `[B]`             | 各窗口 padding 前实际返回的时间点数          |
| `batch['target']['label']`         | ndarray，`int64`     | `[B]`             | 用餐/非用餐/未知标签                         |
| `batch['target']['label_mask']`    | ndarray，`bool`      | `[B]`             | 该标签是否可参与监督损失                     |
| `batch['target']['meal_fraction']` | ndarray，`float32`   | `[B]`             | 与可信用餐区间并集的交叠比例，范围 0–1       |

启用 `window_zscore` 后，三路 `signals` 的 dtype 均变为 `float32`，其他字段类型不变。保留原始 int64 是为了避免 PPG 大整数在直接转 float32 时丢失低位精度。

三种模态没有被强行重采样，`Tacc` 和 `Tgyro` 也不保证相等。批次之间的 `T` 可以变化；最后一个批次的 `B` 可以小于 batch_size。完全缺失的可选模态使用至少一个占位时间点，其 lengths 为 0、valid_mask 全为 False。

channel_mask 根据整个当前 TXT 的非零观测计算，不是设备协议确认的通道启用标志，也不能单独证明该窗口有该模态。应结合 valid_mask 和 lengths 使用。数值 0 可能是真实值或 padding，不能仅凭数值判断有效性。

例如实际 30 秒窗口的一批形状可能为：

```python
batch["signals"]["acc"].shape   # (4, 3, 3203)
batch["signals"]["gyro"].shape  # (4, 3, 3203)
batch["signals"]["ppg"].shape   # (4, 44, 750)
```

这些只是实测示例，不能作为固定输入长度写死。若后续模型需要固定长度或六轴 IMU，需要另行定义时间恢复、对齐与重采样策略。

### meta 字段

`batch['meta']` 是长度为 B 的 Python 列表，每个元素是字典：

| 字段                               | Python 类型      | 含义                                                               |
| ---------------------------------- | ---------------- | ------------------------------------------------------------------ |
| `healthid`                         | `str`            | 平台用户 ID                                                        |
| `subject_group`                    | `str`            | 防止跨集合泄漏的分组 ID；自动分组不等于已核实身份                  |
| `externalid`                       | `str`            | 传感器记录中的外部编号原值                                         |
| `recording_id`                     | `str`            | 传感器记录 uniqueid                                                |
| `zip_name`                         | `str`            | 原始 ZIP 文件名                                                    |
| `member`                           | `str`            | ZIP 内 TXT 成员路径                                                |
| `window_start_ms`、`window_end_ms` | `int`            | 窗口起止 Unix 毫秒时间，左闭右开                                   |
| `time_basis`                       | `str`            | 当前为`observed_batch`，表示原始批次时间                           |
| `meal_ids`                         | `list[str]`      | 与窗口交叠的用餐记录 ID，可能包含不可信记录                        |
| `profile`                          | `dict` 或 `None` | 窗口开始前最近一条资料；原 CSV 字段为字符串，新增 record_ms 为 int |
| `quality_flags`                    | `list[str]`      | 如 no_prior_profile、ppg_missing_or_discontinuous                  |

`meta` 用于关联与溯源，不会自动成为模型输入。

### 标签含义

| label | label_mask | 含义                                                 |
| ----: | ---------- | ---------------------------------------------------- |
|   `1` | `True`     | 窗口完全被可信用餐区间覆盖，且没有不可信用餐交叠     |
|   `0` | `True`     | 已明确确认该用户标注完整，且窗口不与任何用餐区间交叠 |
|  `-1` | `False`    | 未知、用餐边界、可疑用餐或缺少可信标注               |

默认没有确认用餐标注完整，因此不会自动产生 label=0。仅在核实后，将对应 healthid 配置到 `LoaderConfig(complete_annotation_healthids=(...))` 中。不能把 -1 当作非用餐，也不能把 meal_fraction=0 自动当作可信负例。

用餐起止取 `timeFrame`；`beforeTime/afterTime` 是餐前/餐后图片时间，不作为标签边界。默认不足 60 秒的用餐标为不可信，可通过 `build_catalog(..., min_meal_seconds=...)` 修改。

## 6. 空间占用与使用边界

加载器一次处理一个 TXT，将 ZIP 内容流式解压解析为临时二进制数组，不保存额外 TXT 副本，也不建立全量永久信号缓存。临时叶目录使用原 ZIP 文件名去掉 `.zip`，完成后自动删除内容；空的 `.sensor_loader_tmp` 根目录可以保留。

```text
.sensor_loader_tmp/sensor_loader_<随机>/
└── record_<随机>/sensorData-.../
    ├── acc.bin
    ├── gyro.bin
    └── ppg.bin
```

默认预算为每个 loader 的 2 GiB 临时数组，超限会报错并清理，当前没有超限后的自动降级。批次数据另外占用内存，过大的 batch_size 会增加 padding 开销。当前没有后台 worker；多进程分片和详细资源规则见 [加载器说明](sensor_loader/README.md)。

第一批需要等待当前 TXT 解析完成，每次重新遍历会再次解压解析。时间戳保留重复批次时间，不推测逐样本时刻；默认跨越 1 秒以上的主模态缺口时分段。不跨 TXT/ZIP 拼接窗口，也不自动去重不同 ZIP 的重叠采样。

## 7. 验证

运行功能测试：

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s -m unittest discover -s tests -p test_sensor_loader.py -v
```

测试包括 dataTables 布局、孤立辅助记录忽略、时间戳与数据精度、标签、缺失模态、空间限制以及提前退出/异常时清理。字段来源与原始数据分析见 [数据结构与设计说明](docs/数据结构与DataLoader设计.md)。
