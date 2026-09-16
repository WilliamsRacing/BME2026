import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from sensor_loader import Catalog, LoaderConfig, SensorDataLoader, ZipWindowDataset, build_catalog, split_healthids
from sensor_loader.dataset import CHANNELS

BASE = 1784621252000


def make_zip(path, malformed=False, unsafe=False, members=1):
    text = io.StringIO()
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerow(["ACC_TIME", "PPG_TIME", "GYRO_TIME"] + CHANNELS["ppg"] + CHANNELS["acc"] + CHANNELS["gyro"])
    for offset in list(range(0, 4000, 100)) + list(range(10000, 14000, 100)):
        for duplicate in range(2):
            t = BASE + offset
            ppg_t = t if offset % 200 == 0 else 0
            ppg = [20000001 + offset + duplicate] * 20 + [0] * 24 if ppg_t else [0] * 44
            writer.writerow([t, ppg_t, t] + ppg + [offset + 1 + duplicate, 2, 3] + [4, 5, 6])
    if malformed:
        text.write("bad\trow\n")
    names = ["../escape.txt"] if unsafe else ["collect_{}.txt".format(i) for i in range(members)]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("info.json", json.dumps({"files": [{"filename": n} for n in names]}))
        for name in names:
            archive.writestr(name, text.getvalue())


def make_catalog(path):
    recording = {"uniqueid": "rec1", "healthid": "u1", "externalid": "P1", "zip_path": str(path)}
    meals = [dict(uniqueid="meal1", start_ms=BASE, end_ms=BASE+2000, label_valid=True),
             dict(uniqueid="meal2", start_ms=BASE+2400, end_ms=BASE+2800, label_valid=True)]
    return Catalog([recording], {"u1": meals}, {}, {"u1": "s1"}, {})


class SensorLoaderTests(unittest.TestCase):
    def setUp(self):
        self.owner = tempfile.TemporaryDirectory()
        self.root = Path(self.owner.name)
        self.path = self.root / "sensorData-synthetic.zip"
        make_zip(self.path)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.cache = self.root / "cache"
        self.cfg = LoaderConfig(window_seconds=1, stride_seconds=1, gap_seconds=.5,
                                chunk_rows=7, shuffle_files=False, shuffle_windows=False)

    def tearDown(self):
        self.owner.cleanup()

    def dataset(self, config=None):
        return ZipWindowDataset(make_catalog(self.path), config or self.cfg)

    def assert_clean(self):
        self.assertEqual(list(self.cache.iterdir()), [])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.digest)

    def test_windows_preserve_duplicates_precision_and_do_not_cross_gap(self):
        ds = self.dataset()
        with SensorDataLoader(ds, batch_size=2, temp_root=self.cache) as loader:
            batches = list(loader)
        self.assertEqual(sum(len(b["meta"]) for b in batches), 6)
        first = batches[0]
        self.assertEqual(first["signals"]["acc"].shape, (2, 3, 20))
        self.assertEqual(first["signals"]["ppg"].dtype, np.int64)
        self.assertEqual(first["signals"]["ppg"][0, 0, 0], 20000001)
        self.assertEqual(first["timestamp_ms"]["acc"][0, 0], first["timestamp_ms"]["acc"][0, 1])
        self.assertNotEqual(first["signals"]["acc"][0, 0, 0], first["signals"]["acc"][0, 0, 1])
        self.assertFalse(first["valid_mask"]["ppg"][:, 20:].any())
        self.assertEqual([v for b in batches for v in b["target"]["label"].tolist()], [1, 1, -1, -1, -1, -1])
        self.assertEqual(ds.stats["zip_opened"], 1)
        self.assertEqual(ds.stats["members_parsed"], 1)
        self.assert_clean()

    def test_early_break_cleanup_and_detached_batch(self):
        with SensorDataLoader(self.dataset(), batch_size=1, temp_root=self.cache) as loader:
            for batch in loader:
                self.assertTrue(any(self.cache.rglob("acc.bin")))
                break
        self.assertEqual(int(batch["signals"]["acc"][0, 0, 0]), 1)
        self.assert_clean()

    def test_consumer_exception_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "trainer stopped"):
            with SensorDataLoader(self.dataset(), batch_size=1, temp_root=self.cache) as loader:
                for _ in loader:
                    raise RuntimeError("trainer stopped")
        self.assert_clean()

    def test_budget_enforced_and_cleanup(self):
        ds = self.dataset(replace(self.cfg, max_temp_bytes=100))
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            with SensorDataLoader(ds, temp_root=self.cache) as loader:
                list(loader)
        self.assert_clean()

    def test_bad_rows_cleanup(self):
        make_zip(self.path, malformed=True)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "Malformed TSV"):
            with SensorDataLoader(self.dataset(), temp_root=self.cache) as loader:
                list(loader)
        self.assert_clean()

    def test_unsafe_member_rejected(self):
        make_zip(self.path, unsafe=True)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "Unsafe ZIP"):
            with SensorDataLoader(self.dataset(), temp_root=self.cache) as loader:
                list(loader)
        self.assert_clean()

    def test_multiple_members(self):
        make_zip(self.path, members=2)
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        ds = self.dataset()
        with SensorDataLoader(ds, temp_root=self.cache) as loader:
            count = sum(len(b["meta"]) for b in loader)
        self.assertEqual(count, 12)
        self.assertEqual(ds.stats["members_parsed"], 2)
        self.assert_clean()

    def test_negative_labels_require_explicit_annotation_completeness(self):
        ds = self.dataset(replace(self.cfg, complete_annotation_healthids=("u1",)))
        with SensorDataLoader(ds, temp_root=self.cache) as loader:
            labels = [x for b in loader for x in b["target"]["label"].tolist()]
        self.assertEqual(labels, [1, 1, -1, 0, 0, 0])
        self.assert_clean()

    def test_profile_asof_and_overlapping_meals_union(self):
        ds = self.dataset()
        ds.catalog.meals["u1"].append(dict(uniqueid="overlap", start_ms=BASE, end_ms=BASE+2000, label_valid=True))
        ds.catalog.profiles["u1"] = [dict(record_ms=BASE-1, uniqueid="past"), dict(record_ms=BASE+500, uniqueid="future")]
        with SensorDataLoader(ds, batch_size=1, temp_root=self.cache) as loader:
            batch = next(iter(loader))
        self.assertEqual(batch["meta"][0]["profile"]["uniqueid"], "past")
        self.assertEqual(float(batch["target"]["meal_fraction"][0]), 1.0)
        self.assert_clean()

    def test_split_groups_never_cross_sets(self):
        catalog = make_catalog(self.path)
        catalog.groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3", "e": "g4"}
        splits = split_healthids(catalog)
        self.assertEqual(sorted(h for group in splits.values() for h in group), sorted(catalog.groups))
        self.assertEqual(next(k for k,v in splits.items() if "a" in v), next(k for k,v in splits.items() if "b" in v))

    def test_normalization_does_not_lose_ppg_low_bits_before_centering(self):
        ds = self.dataset(replace(self.cfg, normalization="window_zscore"))
        with SensorDataLoader(ds, batch_size=1, temp_root=self.cache) as loader:
            batch = next(iter(loader))
        values = batch["signals"]["ppg"][0, 0]
        self.assertEqual(values.dtype, np.float32)
        self.assertNotEqual(values[0], values[1])
        self.assertAlmostEqual(float(values.mean()), 0, places=6)
        self.assert_clean()

    def test_two_shards_read_disjoint_files(self):
        catalog = make_catalog(self.path)
        second = dict(catalog.recordings[0], uniqueid="rec2")
        catalog.recordings.append(second)
        found = []
        for shard in (0, 1):
            ds = ZipWindowDataset(catalog, self.cfg, shard_id=shard, num_shards=2)
            with SensorDataLoader(ds, temp_root=self.cache) as loader:
                found.append({m["recording_id"] for b in loader for m in b["meta"]})
        self.assertEqual(found, [{"rec1"}, {"rec2"}])
        self.assert_clean()

    def test_sensor_driven_catalog_in_dataTables_ignores_orphans_and_table2(self):
        tables = self.root / "dataTables"
        tables.mkdir()

        def save(name, rows):
            with (tables / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        save("传感器原始数据.csv", [{"uniqueid": "r1", "healthid": "u1", "externalid": "P1",
                                    "timeStamp": repr({"startTime": BASE, "endTime": BASE+14000}),
                                    "sensorData": "kitattachments/" + self.path.name}])
        meal = {"uniqueid": "m1", "healthid": "u1", "externalid": "P1",
                "timeFrame": repr({"startTime": BASE, "endTime": BASE+1000})}
        save("用餐信息1.csv", [meal, dict(meal, uniqueid="orphan", healthid="u2"),
                             dict(meal, uniqueid="outside", timeFrame=repr({"startTime": BASE+20000, "endTime": BASE+21000}))])
        profile = {"uniqueid": "p1", "healthid": "u1", "externalid": "P1", "recordtime": str(BASE-1)}
        save("用户信息.csv", [profile, dict(profile, uniqueid="p2", healthid="u2")])
        (tables / "用餐信息2.csv").write_text("deliberately truncated", encoding="utf-8")
        catalog = build_catalog(self.root, zip_dir=self.root, min_meal_seconds=0)
        self.assertEqual(len(catalog.recordings), 1)
        self.assertEqual(list(catalog.meals), ["u1"])
        self.assertEqual(len(catalog.meals["u1"]), 1)
        self.assertEqual(list(catalog.profiles), ["u1"])
        self.assertEqual(catalog.summary["ignored_meals_without_sensor_user"], 1)
        self.assertEqual(catalog.summary["ignored_meals_without_sensor_time"], 1)
        self.assertEqual(catalog.summary["ignored_profiles_without_sensor_user"], 1)

    def test_missing_optional_modality_is_masked_and_required_one_skips(self):
        # Rewrite the fixture with zero PPG timestamps/values.
        with zipfile.ZipFile(self.path) as archive:
            metadata = archive.read("info.json")
            rows = list(csv.reader(io.StringIO(archive.read("collect_0.txt").decode()), delimiter="\t"))
        for row in rows[1:]:
            row[1] = "0"
            row[3:47] = ["0"] * 44
        text = io.StringIO()
        csv.writer(text, delimiter="\t", lineterminator="\n").writerows(rows)
        with zipfile.ZipFile(self.path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("info.json", metadata)
            archive.writestr("collect_0.txt", text.getvalue())
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with SensorDataLoader(self.dataset(), batch_size=2, temp_root=self.cache) as loader:
            batch = next(iter(loader))
        self.assertEqual(batch["signals"]["ppg"].shape, (2, 44, 1))
        self.assertFalse(batch["valid_mask"]["ppg"].any())
        ds = self.dataset(replace(self.cfg, required_modalities=("acc", "ppg")))
        with SensorDataLoader(ds, temp_root=self.cache) as loader:
            self.assertEqual(list(loader), [])
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
