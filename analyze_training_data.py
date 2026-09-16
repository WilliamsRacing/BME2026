"""Read-only profiling of the four exports and the supplied sensor sample."""
import ast
import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def read_csv(name):
    raw = (ROOT / name).read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode(encoding))))
            return rows, encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("Cannot decode " + name)


def ms(value):
    if value.isdigit():
        return int(value)
    return round(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp() * 1000)


def interval(row, key):
    item = ast.literal_eval(row[key])
    return int(item["startTime"]), int(item["endTime"])


def stats(values):
    a = np.asarray(values)
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())} if len(a) else {}


def union_duration(intervals):
    total, end = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def main():
    tables, result = {}, {}
    for name in ("传感器原始数据.csv", "用餐信息1.csv", "用餐信息2.csv", "用户信息.csv"):
        rows, encoding = read_csv(name)
        tables[name] = rows
        fields = list(rows[0])
        cats = ("deviceType", "sensorType", "recordschema", "dietaryType", "satiety", "tablewareType",
                "dietaryHand", "wearHand", "gender", "wear", "airbag", "drinkingVolume")
        result[name] = {"rows": len(rows), "encoding": encoding, "columns": fields,
                        "uniqueids": len({r["uniqueid"] for r in rows}),
                        "users": len({r["healthid"] for r in rows}),
                        "externalids": len({r["externalid"] for r in rows}),
                        "missing": {f: sum(not r.get(f) for r in rows) for f in fields},
                        "categories": {f: dict(Counter(r[f] for r in rows)) for f in cats if f in fields}}
        if name == "用户信息.csv":
            result[name]["numeric"] = {f: stats([int(r[f]) for r in rows]) for f in ("height", "weight", "age", "wrist")}
        else:
            key = "timeStamp" if name.startswith("传感") else "timeFrame"
            bounds = [interval(r, key) for r in rows]
            result[name]["duration_seconds"] = stats([(b-a)/1000 for a, b in bounds])
            result[name]["invalid_intervals"] = sum(b <= a for a, b in bounds)
            result[name]["range_local"] = [datetime.fromtimestamp(t/1000, TZ).isoformat() for t in (min(a for a,b in bounds), max(b for a,b in bounds))]

    m1 = {r["uniqueid"]: r for r in tables["用餐信息1.csv"]}
    m2 = {r["uniqueid"]: r for r in tables["用餐信息2.csv"]}
    overlap = m1.keys() & m2.keys()
    differences = Counter()
    for key in overlap:
        for f in m1[key]:
            a, b = m1[key][f], m2[key][f]
            if f in ("recordtime", "uploadtime", "beforeTime", "afterTime"):
                # Formatted exports lose millisecond precision.
                a, b = ms(a) // 1000, ms(b) // 1000
            if a != b:
                differences[f] += 1
    result["meals_comparison"] = {"overlap": len(overlap), "only_1": len(m1.keys()-m2.keys()),
                                   "only_2": len(m2.keys()-m1.keys()), "differences_after_time_normalization": dict(differences)}
    sensors = tables["传感器原始数据.csv"]
    meals = list({**m2, **m1}.values())
    profiles = tables["用户信息.csv"]
    by_user = defaultdict(list)
    for r in sensors:
        by_user[r["healthid"]].append(interval(r, "timeStamp"))
    cover = Counter()
    fractions = []
    for r in meals:
        a, b = interval(r, "timeFrame")
        hits = [(max(a,c), min(b,d)) for c,d in by_user[r["healthid"]] if c < b and d > a]
        duration = union_duration(hits)
        if b <= a:
            cover["invalid"] += 1
        else:
            cover["full" if duration == b-a else "partial" if duration else "none"] += 1
            fractions.append(duration/(b-a))
    sensor_users = {r["healthid"] for r in sensors}
    meal_users = {r["healthid"] for r in meals}
    profile_users = {r["healthid"] for r in profiles}
    mapping = defaultdict(set)
    reverse_mapping = defaultdict(set)
    for rows in tables.values():
        for r in rows:
            mapping[r["healthid"]].add(r["externalid"])
            reverse_mapping[r["externalid"]].add(r["healthid"])
    overlaps = 0
    for intervals in by_user.values():
        prev_end = -1
        for a,b in sorted(intervals):
            overlaps += a < prev_end
            prev_end = max(prev_end, b)
    result["relationships"] = {"all_three_users": len(sensor_users & meal_users & profile_users),
                              "sensor_without_meals": len(sensor_users-meal_users),
                              "sensor_without_profile": len(sensor_users-profile_users),
                              "meal_without_sensor": len(meal_users-sensor_users),
                              "healthid_multiple_externalids": sum(len(x)>1 for x in mapping.values()),
                              "externalid_multiple_healthids": {k:len(v) for k,v in reverse_mapping.items() if len(v)>1},
                              "sensor_paths_unique": len({r['sensorData'] for r in sensors}),
                              "sensor_overlapping_intervals": overlaps,
                              "sensor_sum_hours": sum(b-a for v in by_user.values() for a,b in v)/3600000,
                              "sensor_union_hours": sum(union_duration(v) for v in by_user.values())/3600000,
                              "meal_sensor_coverage": dict(cover), "meal_coverage_fraction": stats(fractions),
                              "test_rows": {n:sum(r['externalid'].lower()=='test' for r in rows) for n,rows in tables.items()}}
    repeated_profiles = defaultdict(list)
    for r in profiles:
        repeated_profiles[r["healthid"]].append(r)
    result["profile_revisions"] = {"users_with_multiple_rows": sum(len(v)>1 for v in repeated_profiles.values()),
        "conflicts": {f:sum(len({r[f] for r in rows})>1 for rows in repeated_profiles.values())
                      for f in ("height", "weight", "age", "gender", "wear", "wrist", "airbag")}}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    for path in sorted((ROOT / "样板").glob("*.txt")):
        frame = pd.read_csv(path, sep="\t")
        summary = {"file": path.name, "rows": len(frame), "columns": list(frame.columns),
                   "missing_values": int(frame.isna().sum().sum()), "modalities": {}}
        for name, cols in (("ACC", ["ACC_X", "ACC_Y", "ACC_Z"]),
                           ("GYRO", ["GYRO_X", "GYRO_Y", "GYRO_Z"]),
                           ("PPG", ["PPG"+str(i) for i in range(1,45)])):
            t = frame[name+"_TIME"].to_numpy(dtype=np.int64)
            x = frame[cols].to_numpy()
            nz = t > 0
            valid_t = t[nz]
            starts = np.r_[0, np.flatnonzero(np.diff(valid_t) != 0)+1]
            runs = np.diff(np.r_[starts,len(valid_t)])
            diffs = np.diff(valid_t[starts])
            summary["modalities"][name] = {
                "zero_timestamp_rows": int((~nz).sum()), "valid_timestamp_rows": int(nz.sum()),
                "zero_signal_rows": int((x == 0).all(axis=1).sum()),
                "nonzero_signal_with_zero_timestamp": int(((x != 0).any(axis=1) & ~nz).sum()),
                "all_zero_channels": [c for i,c in enumerate(cols) if (x[:,i] == 0).all()],
                "minmax_each_channel": {c:[float(x[:,i].min()),float(x[:,i].max())] for i,c in enumerate(cols)},
                "first_last_timestamp": [int(valid_t[0]),int(valid_t[-1])] if len(valid_t) else [],
                "time_span_seconds": int(valid_t[-1]-valid_t[0])/1000 if len(valid_t) else None,
                "run_length_counts_top": Counter(runs.tolist()).most_common(10),
                "timestamp_step_ms_top": Counter(diffs.tolist()).most_common(10),
                "backwards_steps": int((diffs<0).sum()), "gaps_gt_1s": int((diffs>1000).sum()),
                "positive_steps": stats(diffs[diffs>0]),
                "first_nonzero_timestamp_row": int(np.flatnonzero(nz)[0]) if nz.any() else None,
            }
        summary["acc_gyro_timestamp_equal_rows"] = int((frame.ACC_TIME == frame.GYRO_TIME).sum())
        start, end = [int(x)*1000 for x in path.stem.split('_')[-2:]]
        summary["matching_sensor_records"] = [{"uniqueid":r["uniqueid"],"externalid":r["externalid"],
                                                "timeStamp":r["timeStamp"],"sensorData":r["sensorData"]}
                                               for r in sensors if interval(r,"timeStamp") == (start,end)]
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
