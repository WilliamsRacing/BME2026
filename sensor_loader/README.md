# 传感器 ZIP DataLoader

只依赖 NumPy，不依赖 PyTorch、TensorFlow 或 GPU。本机用于数据读取和验证；返回的 NumPy 批次可在训练机器上转换为相应框架的张量。

## 环境与快速运行

已在 `D:\anaconda\envs\IMUPPG` 的 Python 3.8.20 中安装 NumPy 1.24.4，没有安装 PyTorch。既有下载 SDK 及其依赖保持原版本。

在项目根目录读取第一个非测试 ZIP 的两个批次：

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py
```

绝对解释器路径避免误用 base 或其他环境，`-s` 禁用用户级 site-packages。数据表默认从 `dataTables/` 读取；没有该目录时兼容项目根目录。也可给 build_catalog 传入 table_dir。ZIP 目录只有一个候选时自动发现，否则使用 zip_dir 显式指定。

本机重新安装依赖：

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -m pip --isolated install --no-cache-dir -r .\requirements-dataloader.txt
```

NumPy 1.24.4 支持 Python 3.8–3.11，见 [NumPy 发布说明](https://numpy.org/doc/2.0/release/1.24.4-notes.html)。迁移至更新的 Python 时应选择兼容的 NumPy 版本，本项目不要求安装训练框架。

读取样板对应的完整 ZIP：

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py --recording-id F0:FA:C7:49:6A:8B_1784621258_1 --max-batches 0
```

读取全部非测试文件（会逐个解压解析，需要时间，不会同时解压全部文件）：

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s .\demo_dataloader.py --max-files 0 --max-batches 0
```

demo 默认只读取一个文件、两个批次，以上两个 0 分别解除限制。可用 `--normalize window_zscore` 输出窗口内标准化后的 float32，默认返回原始 int64。

## 接入程序

```python
from pathlib import Path
from sensor_loader import (
    build_catalog, split_healthids, LoaderConfig,
    ZipWindowDataset, SensorDataLoader,
)

root = Path(r"D:\William\code\BME\BME2026")
catalog = build_catalog(root)
splits = split_healthids(catalog, seed=42)
config = LoaderConfig(
    window_seconds=30,
    stride_seconds=15,
    gap_seconds=1,
    required_modalities=("acc",),
    ppg_channels=tuple(range(1, 45)),
    max_temp_bytes=2 * 1024**3,
    normalization="none",
)
dataset = ZipWindowDataset(catalog, config, healthids=splits["train"])

for epoch in range(2):
    dataset.set_epoch(epoch)
    with SensorDataLoader(
        dataset, batch_size=16, temp_root=root / ".sensor_loader_tmp"
    ) as loader:
        for batch in loader:
            acc = batch["signals"]["acc"]
            gyro = batch["signals"]["gyro"]
            ppg = batch["signals"]["ppg"]
            target = batch["target"]["label"]
            target_mask = batch["target"]["label_mask"]
            # 在训练机器上把这些 NumPy 数组交给训练器。
            # 监督损失只使用 target_mask=True 的样本。
```

必须使用 `with`。正常结束、提前 break、消费者异常、文件解析失败或临时空间超限都会释放映射并清理本次读取器的临时文件。每个 epoch 按 seed + epoch 改变文件和窗口顺序。

没有精确的 `len(dataset)`：有效窗口数依赖 ZIP 内采样中断，不能仅凭 CSV 时长得出。完整读取后可查看 `dataset.stats['windows']`；需要预先确定窗口数时，应扫描所用文件或使用按迭代步数运行的训练器。

## 返回批次

| 键 | 类型/形状 | 说明 |
| --- | --- | --- |
| signals.acc | `[B, 3, Tacc]` | 三轴加速度 |
| signals.gyro | `[B, 3, Tgyro]` | 三轴陀螺仪 |
| signals.ppg | `[B, Cppg, Tppg]` | 默认 44 个 PPG 字段，可选固定子集 |
| valid_mask.* | bool，同对应 signals 形状 | 有效样本位置与通道，padding 为 False |
| channel_mask.* | bool，`[B, C]` | 当前 TXT 中该通道是否出现非零值；是观测指标，不是设备通道启用标志 |
| lengths.* | int64，`[B]` | padding 前样本数 |
| timestamp_ms.* | int64，`[B, T]` | 原始批次时间戳，允许重复，padding 为 0 |
| relative_time_s.* | float32，`[B, T]` | 相对窗口起点秒数，结合 mask 使用 |
| target.label | int64，`[B]` | 1 用餐、0 可信非用餐、-1 未知/边界 |
| target.label_mask | bool，`[B]` | 是否可用于监督损失 |
| target.meal_fraction | float32，`[B]` | 与可信用餐区间并集的交叠比例；无标注时的 0 不是负例真值 |
| meta | 长度 B 的列表 | 记录 ID、用户、窗口范围、ZIP/TXT 名、标签来源、资料和质量标记 |

三路数据分别 padding，长度会随批次变化；完全缺失的可选模态返回一个全部被 mask 的占位位置。ACC/GYRO 没有被按行强行拼成六轴 IMU，因为真实文件存在两者采样数不同的情况。若模型需要六轴等长输入，应在确认采样时间协议后增加对齐变换。

默认 int64 保留原始 PPG 大整数精度。`window_zscore` 使用 float64 居中/缩放后转 float32，不用全数据拟合参数；它会去掉窗口的绝对幅度/基线，是否适合任务由训练方案决定。模型输入不自动包括 meta 中的身份、餐次或资料。

## 关联与标签

- 传感器 CSV 是主表，辅助记录没有对应传感器数据时忽略。只读完整用餐表 1，不重复合并表 2。
- 默认忽略 externalid 为 test 的两条传感器记录。保留全部 1165 条可用 `build_catalog(root, exclude_test=False)`；demo 对应 `--include-test-records`。
- 缺少 ZIP 的记录不进入可读清单，计入 catalog.summary 的 missing_zip_rows。坏 ZIP/TSV、时钟回退、空间超限明确报错，不静默跳过。
- 不足 60 秒的用餐默认标为不可信，用 `build_catalog(..., min_meal_seconds=...)` 调整；保留其区间，避免错误生成负样本。
- 用餐标签按相同 healthid 下的 timeFrame 关联，区间为 `[start, end)`。窗口完全被可信用餐覆盖时 label=1；部分覆盖、可疑用餐、标注缺失默认 -1。
- 只有已确认用餐标注完整的用户，才设置 `complete_annotation_healthids=(...)`；这些用户不与任何用餐交叠的窗口才能标为 0。默认不会产生可信负样本。
- 用户资料选窗口起点之前最近一条。没有历史资料时 meta.profile 为 None，不使用未来资料，也不删除信号。

当前目录默认保留 1163 条非测试传感器记录、40 个 healthid、274 条有传感器声明范围交叠的用餐、53 条资料。274 条用餐中 6 条过短而被标为不可信。最终标签仍需要附件内部真实信号窗口。

`split_healthids` 按用户组划分，不随机拆分 ZIP/窗口。默认保守合并保留记录中共享外部编号的 healthid，不能自动识别多人复用同一账户。核对后的 healthid -> subject_id 映射可传入 build_catalog 的 subject_mapping。默认比例 70%/15%/15%，seed=42，本批数据得到 28/6/6 个 healthid；正式实验应保存并核对划分。

## 空间控制与清理

读取 info.json 后逐个流式解压其声明的 TXT，分块解析为三路临时二进制数组，不额外保存 TXT，不永久缓存全量信号，支持一个 ZIP 多个 TXT。

```text
原附件目录/sensorData-....zip              # 始终只读保留
.sensor_loader_tmp/sensor_loader_<随机>/
    record_<随机>/sensorData-.../           # 叶目录为 ZIP 名（去掉 .zip）
        acc.bin
        gyro.bin
        ppg.bin
```

每次只保留一个 TXT 的解析数组，用完或退出时删除。原 ZIP 留在原处，无需重新压缩。批次返回独立内存副本，清理文件后仍然有效。

默认临时数组预算 2 GiB，写入前检查，超限报错并清理。本版本不提供超限时的自动降级，需要显式增加预算或处理源文件。内存受解析分块、单窗口最大样本数和 batch_size 约束，padding 较大时仍会占较多内存，建议从 batch_size=4 或 16 开始。

单进程读取，不自动创建后台 worker。训练机器如需多进程，可分别使用 `ZipWindowDataset(..., shard_id=i, num_shards=n)`，保证相同输入清单和 seed/epoch，文件分片互不重叠，各进程独立创建 loader 上下文。**预算按每个 loader 计算**，n 个并发实例总预算最多 n 倍；希望总量为 2 GiB 时分别设置为 2 GiB/n。

同一文件内生成多个窗口，不会每个窗口重新解压。每轮再次访问文件仍需解压解析，首批也需等待当前 TXT 解析完，是节省长期磁盘占用的取舍。

正常异常路径均清理；操作系统强杀或断电可能遗留独占临时目录。此版本不自动删除其他运行的目录，确认相应进程已结束后才手动清理残留。

## 时间精度与边界

- 使用观测批次时间，保留重复时间戳；不猜测首/末样本语义或逐样本采样率。meta.time_basis 为 observed_batch。
- 默认按 ACC 时间间隔超过 1 秒分段，窗口不跨中断，可配置 gap_seconds。
- 可选模态在窗口内中断或两端缺失超阈值时置为缺失；必需模态不满足条件则跳过整个窗口。
- 时间边界精度受批时间语义限制，适合数据读取和用餐区间级原型，不能声称实现了逐口动作级精确同步。
- 不跨 TXT/ZIP 拼接，也不自动去重不同 ZIP 中的重叠采样；如实验需要，须核对设备和重叠样本后增加处理。
- 物理单位、PPG 字段映射、时钟含义仍需采集协议确认，代码未猜测这些信息。

## 验证

```powershell
& 'D:\anaconda\envs\IMUPPG\python.exe' -s -m unittest discover -s tests -p test_sensor_loader.py -v
```

覆盖重复时间戳、整数精度、间断、标签并集、资料历史匹配、分组划分、文件分片、多个 TXT、提前退出、异常、损坏行、路径和空间预算。

真实样板完整验证：30 秒窗口/15 秒步长生成 278 个窗口，其中 100 个可信用餐、178 个未知；临时数组峰值 68,591,368 字节。退出后临时目录为空，ZIP SHA-256 前后一致。另外在文件大小排序中选择最小、中位和 75% 位置的三个真实 ZIP，均成功生成批次并清理。
