"""Sensor-driven metadata joins. Source CSVs are never modified."""
import ast
import csv
import hashlib
import io
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))


def read_rows(path):
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unsupported CSV encoding: {}".format(path))
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    for index, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Truncated/malformed CSV record {} in {}".format(index, path))
    return rows


def timestamp_ms(value):
    value = str(value).strip()
    if value.isdigit():
        number = int(value)
        if number < 100000000000:
            raise ValueError("Expected epoch milliseconds, got {}".format(value))
        return number
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp() * 1000)


def time_range(value):
    data = ast.literal_eval(value)
    start, end = int(data["startTime"]), int(data["endTime"])
    if start <= 0 or end <= start:
        raise ValueError("Invalid time range: {}".format(value))
    return start, end


def overlap(a, b, c, d):
    return a < d and c < b


def union_ms(intervals):
    total, last = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, last))
        last = max(last, end)
    return total


@dataclass
class Catalog:
    recordings: list
    meals: dict
    profiles: dict
    groups: dict
    summary: dict


def build_catalog(root, zip_dir=None, exclude_test=True, min_meal_seconds=60,
                  subject_mapping=None, table_dir=None):
    """Ignore auxiliary rows without sensors; retain unmatched sensor windows.

    `subject_mapping`, if supplied, must map every retained healthid to a verified
    subject ID. Without it, groups conservatively connect matching external IDs.
    These automatic groups prevent leakage but are not verified identities.
    """
    root = Path(root).resolve()
    if zip_dir is None:
        candidates = [p for p in root.iterdir() if p.is_dir() and next(p.glob("sensorData-*.zip"), None)]
        if len(candidates) != 1:
            raise ValueError("Pass zip_dir explicitly; found {} candidate directories".format(len(candidates)))
        zip_dir = candidates[0]
    zip_dir = Path(zip_dir).resolve()
    if not zip_dir.is_dir():
        raise FileNotFoundError(zip_dir)
    if table_dir is None:
        table_dir = root / "dataTables" if (root / "dataTables").is_dir() else root
    table_dir = Path(table_dir).resolve()
    sensors = read_rows(table_dir / "传感器原始数据.csv")
    meals = read_rows(table_dir / "用餐信息1.csv")
    profiles = read_rows(table_dir / "用户信息.csv")
    # Meal table 2 is a truncated subset, deliberately not read or concatenated.
    counts = Counter(sensor_rows=len(sensors), meal_rows=len(meals), profile_rows=len(profiles))
    recordings, seen = [], set()
    for row in sensors:
        if row["uniqueid"] in seen:
            raise ValueError("Duplicate sensor uniqueid: " + row["uniqueid"])
        seen.add(row["uniqueid"])
        if exclude_test and row["externalid"].strip().lower() == "test":
            counts["ignored_test_sensor_rows"] += 1
            continue
        if not row["healthid"]:
            raise ValueError("Empty sensor healthid")
        start, end = time_range(row["timeStamp"])
        basename = row["sensorData"].replace("\\", "/").rsplit("/", 1)[-1]
        if not basename.lower().endswith(".zip") or basename in (".zip", "..zip"):
            raise ValueError("Invalid sensor ZIP name")
        path = zip_dir / basename
        if not path.is_file():
            counts["missing_zip_rows"] += 1
            continue
        recordings.append(dict(row, start_ms=start, end_ms=end, zip_path=str(path)))
    healthids = {r["healthid"] for r in recordings}
    sensor_ranges = defaultdict(list)
    for row in recordings:
        sensor_ranges[row["healthid"]].append((row["start_ms"], row["end_ms"]))
    meals_by_user, profiles_by_user = defaultdict(list), defaultdict(list)
    seen_meals = set()
    for row in meals:
        if row["uniqueid"] in seen_meals:
            raise ValueError("Duplicate meal uniqueid")
        seen_meals.add(row["uniqueid"])
        if row["healthid"] not in healthids:
            counts["ignored_meals_without_sensor_user"] += 1
            continue
        a, b = time_range(row["timeFrame"])
        if not any(overlap(a, b, c, d) for c, d in sensor_ranges[row["healthid"]]):
            counts["ignored_meals_without_sensor_time"] += 1
            continue
        valid = (b - a >= min_meal_seconds * 1000 and row["externalid"].strip().lower() != "test")
        meals_by_user[row["healthid"]].append(dict(row, start_ms=a, end_ms=b, label_valid=valid))
        counts["kept_meals"] += 1
        counts["untrusted_meals"] += int(not valid)
    for row in profiles:
        if row["healthid"] not in healthids:
            counts["ignored_profiles_without_sensor_user"] += 1
            continue
        profiles_by_user[row["healthid"]].append(dict(row, record_ms=timestamp_ms(row["recordtime"])))
    for rows in profiles_by_user.values():
        rows.sort(key=lambda row: (row["record_ms"], row["uniqueid"]))
    for rows in meals_by_user.values():
        rows.sort(key=lambda row: (row["start_ms"], row["uniqueid"]))

    if subject_mapping is not None:
        if healthids - subject_mapping.keys():
            raise ValueError("subject_mapping must include all retained healthids")
        groups = {h: str(subject_mapping[h]) for h in healthids}
    else:
        parent = {h: h for h in healthids}

        def find(h):
            while parent[h] != h:
                parent[h] = parent[parent[h]]
                h = parent[h]
            return h

        aliases = {}
        relevant_rows = recordings + [r for rs in meals_by_user.values() for r in rs] + [r for rs in profiles_by_user.values() for r in rs]
        for row in relevant_rows:
            alias = "".join(row["externalid"].split()).upper()
            if not alias or alias == "TEST":
                continue
            h = row["healthid"]
            if alias in aliases:
                parent[find(h)] = find(aliases[alias])
            else:
                aliases[alias] = h
        members = defaultdict(list)
        for h in sorted(healthids):
            members[find(h)].append(h)
        groups = {}
        for group in members.values():
            key = "group_" + hashlib.sha256("|".join(group).encode()).hexdigest()[:16]
            groups.update({h: key for h in group})
    counts["recordings"] = len(recordings)
    counts["healthids"] = len(healthids)
    counts["split_groups"] = len(set(groups.values()))
    counts["kept_profiles"] = sum(len(v) for v in profiles_by_user.values())
    return Catalog(recordings, dict(meals_by_user), dict(profiles_by_user), groups, dict(counts))


def split_healthids(catalog, ratios=(0.7, 0.15, 0.15), seed=42):
    """Deterministic disjoint group splits; no random window/ZIP split."""
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to 1")
    groups = sorted(set(catalog.groups.values()))
    if len(groups) < 3:
        raise ValueError("At least three groups are needed for three splits")
    random.Random(seed).shuffle(groups)
    nval = max(1, round(len(groups) * ratios[1]))
    ntest = max(1, round(len(groups) * ratios[2]))
    ntrain = len(groups) - nval - ntest
    if ntrain < 1:
        raise ValueError("Split ratios leave no training group")
    result = {}
    for name, subset in (("train", groups[:ntrain]), ("val", groups[ntrain:ntrain+nval]), ("test", groups[ntrain+nval:])):
        result[name] = sorted(h for h, group in catalog.groups.items() if group in set(subset))
    return result
