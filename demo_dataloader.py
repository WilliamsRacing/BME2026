"""Read real sensor batches without a training framework or model."""
import argparse
import json
from pathlib import Path

from sensor_loader import LoaderConfig, SensorDataLoader, ZipWindowDataset, build_catalog, split_healthids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--zip-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=30)
    parser.add_argument("--stride-seconds", type=float, default=15)
    parser.add_argument("--max-batches", type=int, default=2, help="0 reads the whole selected dataset")
    parser.add_argument("--max-files", type=int, default=1, help="0 selects all files")
    parser.add_argument("--recording-id", action="append")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--normalize", choices=("none", "window_zscore"), default="none")
    parser.add_argument("--temp-gib", type=float, default=2)
    parser.add_argument("--include-test-records", action="store_true")
    args = parser.parse_args()
    catalog = build_catalog(args.root, args.zip_dir, exclude_test=not args.include_test_records)
    print("Catalog:", json.dumps(catalog.summary, ensure_ascii=False), flush=True)
    users = None if args.split == "all" else split_healthids(catalog)[args.split]
    config = LoaderConfig(window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
                          shuffle_files=False, shuffle_windows=False,
                          normalization=args.normalize, max_temp_bytes=int(args.temp_gib * 1024**3))
    dataset = ZipWindowDataset(catalog, config, healthids=users, recording_ids=args.recording_id)
    if args.max_files:
        dataset.recordings = dataset.recordings[:args.max_files]
    temp_root = args.root / ".sensor_loader_tmp"
    batches = 0
    with SensorDataLoader(dataset, batch_size=args.batch_size, temp_root=temp_root) as loader:
        for batch in loader:
            batches += 1
            print(json.dumps({
                "batch": batches,
                "shapes": {m: list(a.shape) for m, a in batch["signals"].items()},
                "dtypes": {m: str(a.dtype) for m, a in batch["signals"].items()},
                "labels": batch["target"]["label"].tolist(),
                "label_mask": batch["target"]["label_mask"].tolist(),
                "first_source": batch["meta"][0]["zip_name"],
                "first_window_start_ms": batch["meta"][0]["window_start_ms"],
            }, ensure_ascii=False), flush=True)
            if args.max_batches and batches >= args.max_batches:
                break
    print("Read stats:", json.dumps(dataset.stats), flush=True)
    print("Loader closed; its temporary files were removed. Original ZIPs are unchanged.")
    if not batches:
        print("No eligible windows in the selection; check duration, gaps and required modalities.")


if __name__ == "__main__":
    main()
