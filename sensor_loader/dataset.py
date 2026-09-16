"""Bounded temporary storage, native timestamp windows and NumPy batches.

No torch/TensorFlow dependency. Use SensorDataLoader as a context manager so an
early break or exception also closes memory maps and removes temporary files.
"""
import csv
import io
import json
import math
import random
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .catalog import overlap, union_ms

CHANNELS = {
    "acc": ["ACC_X", "ACC_Y", "ACC_Z"],
    "gyro": ["GYRO_X", "GYRO_Y", "GYRO_Z"],
    "ppg": ["PPG{}".format(i) for i in range(1, 45)],
}


@dataclass(frozen=True)
class LoaderConfig:
    window_seconds: float = 30.0
    stride_seconds: float = 15.0
    gap_seconds: float = 1.0
    primary_modality: str = "acc"
    required_modalities: tuple = ("acc",)
    ppg_channels: tuple = tuple(range(1, 45))
    chunk_rows: int = 4096
    max_temp_bytes: int = 2 * 1024 ** 3
    max_samples_per_window: int = 100000
    shuffle_files: bool = True
    shuffle_windows: bool = True
    seed: int = 42
    normalization: str = "none"
    # Only explicitly verified complete annotation allows negative labels.
    complete_annotation_healthids: tuple = ()

    def __post_init__(self):
        for value in (self.window_seconds, self.stride_seconds, self.gap_seconds):
            if not math.isfinite(value) or value * 1000 < 1:
                raise ValueError("Time settings must be finite and at least 1 ms")
        if self.primary_modality not in CHANNELS:
            raise ValueError("Unknown primary modality")
        if any(m not in CHANNELS for m in self.required_modalities):
            raise ValueError("Unknown required modality")
        if not self.ppg_channels or len(set(self.ppg_channels)) != len(self.ppg_channels) or any(i not in range(1, 45) for i in self.ppg_channels):
            raise ValueError("ppg_channels must be unique integers in 1..44")
        if self.chunk_rows < 1 or self.max_temp_bytes < 1 or self.max_samples_per_window < 1:
            raise ValueError("Resource limits must be positive")
        if self.normalization not in ("none", "window_zscore"):
            raise ValueError("normalization must be none or window_zscore")


def _safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe ZIP member name: {!r}".format(name))
    return name


def _members(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("ZIP contains duplicate member names")
    infos = [n for n in names if PurePosixPath(n).name == "info.json"]
    if len(infos) != 1:
        raise ValueError("Expected exactly one info.json")
    name = _safe_member(infos[0])
    if archive.getinfo(name).file_size > 1024 * 1024:
        raise ValueError("Oversized info.json")
    metadata = json.loads(archive.read(name).decode("utf-8-sig"))
    base = PurePosixPath(name).parent
    result = []
    for item in metadata["files"]:
        filename = _safe_member(item["filename"])
        member = _safe_member(str(base / filename))
        if not filename.lower().endswith(".txt"):
            continue
        if member not in names:
            raise ValueError("Missing declared TXT: " + member)
        result.append(member)
    if not result or len(result) != len(set(result)):
        raise ValueError("No TXT files or duplicate TXT references")
    return result


@contextmanager
def _prepare_member(archive, member, directory, config):
    """Stream-decompress one TXT directly to bounded binary arrays, not a TXT copy.

    The compression source is read through EOF (ZipExtFile verifies its CRC).
    Each row stores an int64 observed batch timestamp and raw channel values.
    """
    arrays, handles, counts, active, paths = {}, {}, {}, {}, {}
    used = 0
    try:
        for modality, columns in CHANNELS.items():
            paths[modality] = directory / (modality + ".bin")
            handles[modality] = paths[modality].open("wb")
            counts[modality] = 0
            active[modality] = np.zeros(len(columns), dtype=bool)
        last = {m: None for m in CHANNELS}
        with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            reader = csv.reader(text, delimiter="\t")
            header = next(reader, None)
            expected = ["ACC_TIME", "PPG_TIME", "GYRO_TIME"] + CHANNELS["ppg"] + CHANNELS["acc"] + CHANNELS["gyro"]
            if header is None or len(header) != len(set(header)) or set(header) != set(expected):
                raise ValueError("Unexpected sensor columns in " + member)
            indices = {m: [header.index(m.upper() + "_TIME")] + [header.index(c) for c in columns]
                       for m, columns in CHANNELS.items()}
            chunk = []

            def write_chunk(rows):
                nonlocal used
                data = np.asarray(rows, dtype=np.int64)
                if data.ndim != 2 or data.shape[1] != len(header):
                    raise ValueError("Malformed TSV row")
                if np.any(data[:, [header.index(m.upper() + "_TIME") for m in CHANNELS]] < 0):
                    raise ValueError("Negative sensor timestamp")
                for m in CHANNELS:
                    block = np.ascontiguousarray(data[:, indices[m]])
                    block = block[block[:, 0] > 0]
                    if not len(block):
                        continue
                    if (last[m] is not None and block[0, 0] < last[m]) or np.any(np.diff(block[:, 0]) < 0):
                        raise ValueError("Timestamp reversal in {} / {}".format(member, m))
                    last[m] = int(block[-1, 0])
                    if used + block.nbytes > config.max_temp_bytes:
                        raise OSError("Temporary array budget exceeded ({} bytes); increase max_temp_bytes or use smaller source files".format(config.max_temp_bytes))
                    block.tofile(handles[m])
                    used += block.nbytes
                    counts[m] += len(block)
                    active[m] |= np.any(block[:, 1:] != 0, axis=0)

            for line_no, row in enumerate(reader, 2):
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("Malformed TSV at {}:{}".format(member, line_no))
                chunk.append(row)
                if len(chunk) >= config.chunk_rows:
                    write_chunk(chunk)
                    chunk.clear()
            if chunk:
                write_chunk(chunk)
        for handle in handles.values():
            handle.close()
        for m, columns in CHANNELS.items():
            shape = (counts[m], len(columns) + 1)
            arrays[m] = np.memmap(paths[m], dtype=np.int64, mode="r", shape=shape) if counts[m] else np.empty(shape, dtype=np.int64)
        yield arrays, active, used
    finally:
        for array in arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
        for handle in handles.values():
            handle.close()
        for path in paths.values():
            if path.is_file():
                path.unlink()


def _segments(times, gap_ms):
    """Yield index ranges without materializing a whole-file diff array."""
    if not len(times):
        return
    begin = 0
    for start in range(0, len(times), 65536):
        stop = min(len(times), start + 65536)
        if start and times[start] - times[start - 1] > gap_ms:
            yield begin, start
            begin = start
        cuts = np.flatnonzero(np.diff(times[start:stop]) > gap_ms) + start + 1
        for cut in cuts:
            yield begin, int(cut)
            begin = int(cut)
    yield begin, len(times)


class ZipWindowDataset:
    """Framework-independent iterable source; no global in-memory signal cache.

    Window times use observed batch timestamps. Repeated times are preserved;
    no undocumented sample rate or within-batch timing is invented.
    """

    def __init__(self, catalog, config=None, healthids=None, recording_ids=None,
                 shard_id=0, num_shards=1):
        if num_shards < 1 or not 0 <= shard_id < num_shards:
            raise ValueError("Invalid shard assignment")
        self.catalog = catalog
        self.config = config or LoaderConfig()
        allowed_users = set(healthids) if healthids is not None else set(catalog.groups)
        allowed_records = set(recording_ids) if recording_ids is not None else None
        self.recordings = [r for r in catalog.recordings if r["healthid"] in allowed_users and
                           (allowed_records is None or r["uniqueid"] in allowed_records)]
        self.recordings.sort(key=lambda r: r["uniqueid"])
        self.shard_id, self.num_shards, self.epoch = shard_id, num_shards, 0
        self.stats = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def iter_samples(self, temp_root):
        cfg = self.config
        rng = random.Random(cfg.seed + self.epoch)
        recordings = list(self.recordings)
        if cfg.shuffle_files:
            rng.shuffle(recordings)
        # Same seed/epoch on every shard, then partition: disjoint ZIP ownership.
        recordings = recordings[self.shard_id::self.num_shards]
        self.stats = {"zip_opened": 0, "members_parsed": 0, "windows": 0,
                      "skipped_required_modality": 0, "peak_temp_bytes": 0}
        for recording in recordings:
            path = Path(recording["zip_path"])
            # A unique parent permits concurrent independent loader instances.
            with tempfile.TemporaryDirectory(prefix="record_", dir=str(temp_root)) as owner:
                directory = Path(owner) / path.stem
                directory.mkdir()
                try:
                    with zipfile.ZipFile(path) as archive:
                        self.stats["zip_opened"] += 1
                        members = _members(archive)
                        for member in members:
                            with _prepare_member(archive, member, directory, cfg) as (arrays, active, used):
                                self.stats["members_parsed"] += 1
                                self.stats["peak_temp_bytes"] = max(self.stats["peak_temp_bytes"], used)
                                yield from self._member_samples(recording, member, arrays, active, rng)
                except Exception as exc:
                    raise RuntimeError("Failed reading {}: {}".format(path.name, exc)) from exc

    def _member_samples(self, recording, member, arrays, active, rng):
        cfg = self.config
        window_ms, stride_ms, gap_ms = [round(x * 1000) for x in (cfg.window_seconds, cfg.stride_seconds, cfg.gap_seconds)]
        times = arrays[cfg.primary_modality][:, 0]
        segments = list(_segments(times, gap_ms))
        if cfg.shuffle_windows:
            rng.shuffle(segments)
        for lo, hi in segments:
            first, end = int(times[lo]), int(times[hi - 1]) + 1
            count = max(0, (end - first - window_ms) // stride_ms + 1)
            # O(1) permutation storage even for a long recording.
            offset, step = (0, 1)
            if cfg.shuffle_windows and count > 1:
                offset = rng.randrange(count)
                step = rng.randrange(1, count)
                while math.gcd(step, count) != 1:
                    step = rng.randrange(1, count)
            for index in range(count):
                start = first + ((offset + index * step) % count) * stride_ms
                sample = self._sample(recording, member, arrays, active, start, start + window_ms, gap_ms)
                if sample is not None:
                    self.stats["windows"] += 1
                    yield sample

    def _sample(self, recording, member, arrays, active, start, end, gap_ms):
        cfg = self.config
        signals, timestamps, channel_masks, quality = {}, {}, {}, []
        for modality, array in arrays.items():
            t = array[:, 0]
            lo, hi = np.searchsorted(t, [start, end], side="left")
            if hi - lo > cfg.max_samples_per_window:
                raise ValueError("Too many samples in one window; check timestamp units/configuration")
            selected = np.asarray(array[lo:hi])
            continuous = (len(selected) > 0 and selected[0, 0] - start <= gap_ms and
                          end - selected[-1, 0] <= gap_ms and not np.any(np.diff(selected[:, 0]) > gap_ms))
            if not continuous:
                if modality in set(cfg.required_modalities) | {cfg.primary_modality}:
                    self.stats["skipped_required_modality"] += 1
                    return None
                selected = np.empty((0, array.shape[1]), dtype=np.int64)
                quality.append(modality + "_missing_or_discontinuous")
            columns = np.asarray(cfg.ppg_channels, dtype=int) if modality == "ppg" else np.arange(1, array.shape[1])
            raw = selected[:, columns].T.copy()
            if cfg.normalization == "window_zscore":
                data = raw.astype(np.float64)
                if data.shape[1]:
                    data -= data.mean(axis=1, keepdims=True)
                    scale = data.std(axis=1, keepdims=True)
                    data /= np.where(scale > 0, scale, 1.0)
                signals[modality] = data.astype(np.float32)
            else:
                signals[modality] = raw  # int64 preserves all original integer precision
            timestamps[modality] = selected[:, 0].copy()
            channel_masks[modality] = active[modality][columns - 1].copy()

        healthid = recording["healthid"]
        meals = [m for m in self.catalog.meals.get(healthid, []) if overlap(start, end, m["start_ms"], m["end_ms"])]
        covered = union_ms([(max(start, m["start_ms"]), min(end, m["end_ms"])) for m in meals if m["label_valid"]])
        uncertain = any(not m["label_valid"] for m in meals)
        label = -1
        if covered == end - start and not uncertain:
            label = 1
        elif not meals and healthid in cfg.complete_annotation_healthids:
            label = 0
        profile = None
        for row in self.catalog.profiles.get(healthid, []):
            if row["record_ms"] <= start:
                profile = dict(row)
            else:
                break
        if profile is None:
            quality.append("no_prior_profile")
        return {
            "signals": signals,
            "timestamp_ms": timestamps,
            "channel_mask": channel_masks,
            "target": {"label": label, "label_mask": label >= 0,
                       "meal_fraction": covered / (end - start)},
            "meta": {"healthid": healthid, "subject_group": self.catalog.groups[healthid],
                     "externalid": recording["externalid"], "recording_id": recording["uniqueid"],
                     "zip_name": Path(recording["zip_path"]).name, "member": member,
                     "window_start_ms": start, "window_end_ms": end,
                     "time_basis": "observed_batch", "meal_ids": [m["uniqueid"] for m in meals],
                     "profile": profile, "quality_flags": quality},
        }


def collate_samples(samples):
    """Pad native-rate streams independently; all outputs are NumPy/Python."""
    batch = {"signals": {}, "valid_mask": {}, "channel_mask": {},
             "timestamp_ms": {}, "relative_time_s": {}, "lengths": {}, "meta": []}
    for modality in CHANNELS:
        channels = samples[0]["signals"][modality].shape[0]
        lengths = np.asarray([s["signals"][modality].shape[1] for s in samples], dtype=np.int64)
        # A fully missing modality still has one masked padding position.
        width = max(1, int(lengths.max()))
        shape = (len(samples), channels, width)
        data = np.zeros(shape, dtype=samples[0]["signals"][modality].dtype)
        valid = np.zeros(shape, dtype=bool)
        timestamps = np.zeros((len(samples), width), dtype=np.int64)
        relative = np.zeros((len(samples), width), dtype=np.float32)
        channel_mask = np.stack([s["channel_mask"][modality] for s in samples])
        for i, sample in enumerate(samples):
            n = int(lengths[i])
            data[i, :, :n] = sample["signals"][modality]
            valid[i, :, :n] = channel_mask[i, :, None]
            timestamps[i, :n] = sample["timestamp_ms"][modality]
            relative[i, :n] = (sample["timestamp_ms"][modality] - sample["meta"]["window_start_ms"]) / 1000.0
        batch["signals"][modality] = data
        batch["valid_mask"][modality] = valid
        batch["channel_mask"][modality] = channel_mask
        batch["timestamp_ms"][modality] = timestamps
        batch["relative_time_s"][modality] = relative
        batch["lengths"][modality] = lengths
    batch["target"] = {
        "label": np.asarray([s["target"]["label"] for s in samples], dtype=np.int64),
        "label_mask": np.asarray([s["target"]["label_mask"] for s in samples], dtype=bool),
        "meal_fraction": np.asarray([s["target"]["meal_fraction"] for s in samples], dtype=np.float32),
    }
    batch["meta"] = [s["meta"] for s in samples]
    return batch


class SensorDataLoader:
    """Context-managed batching with one decompressed member at a time.

    A loader has no background workers. External processes may use disjoint
    dataset shard IDs; max_temp_bytes applies per loader/process, not globally.
    """

    def __init__(self, dataset, batch_size=16, temp_root=None, drop_last=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size, self.drop_last = dataset, int(batch_size), drop_last
        self.temp_root = Path(temp_root).resolve() if temp_root else None
        self._owner = None
        self._iterator = None

    def __enter__(self):
        if self._owner is not None:
            raise RuntimeError("Loader is already open")
        if self.temp_root:
            self.temp_root.mkdir(parents=True, exist_ok=True)
        self._owner = tempfile.TemporaryDirectory(prefix="sensor_loader_", dir=str(self.temp_root) if self.temp_root else None)
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        if self._owner is None:
            raise RuntimeError("Use 'with SensorDataLoader(...) as loader:' for deterministic cleanup")
        if self._iterator is not None:
            self._iterator.close()
        self._iterator = self._batches()
        return self._iterator

    def _batches(self):
        source = self.dataset.iter_samples(Path(self._owner.name))
        pending = []
        try:
            for sample in source:
                pending.append(sample)
                if len(pending) == self.batch_size:
                    yield collate_samples(pending)
                    pending.clear()
            if pending and not self.drop_last:
                yield collate_samples(pending)
        finally:
            source.close()

    def close(self):
        try:
            if self._iterator is not None:
                self._iterator.close()
        finally:
            self._iterator = None
            if self._owner is not None:
                self._owner.cleanup()
                self._owner = None


def managed_dataloader(dataset, **kwargs):
    return SensorDataLoader(dataset, **kwargs)
