'''
合并 ScanNet qa_jsonl 文件，生成 all/train/val jsonl 文件，先不管"distance_infer_center_oc"任务[因为没有2d物体标注]
Selected jsonl files: 20
type=depth_prediction_oc total=2988 test_sampled=200
type=depth_prediction_oo total=1971 test_sampled=200
type=distance_infer_center_oo total=1928 test_sampled=200
type=distance_prediction_oc total=2452 test_sampled=200
type=distance_prediction_oo total=2486 test_sampled=200
type=obj_spatial_relation_oo total=6393 test_sampled=200
type=spatial_imagination_oc total=3208 test_sampled=200
type=spatial_imagination_oo total=3229 test_sampled=200
type=spatial_volume_infer total=2457 test_sampled=200
All records: 27112 -> /data/dsq/ScanNet/qa_jsonl/all.jsonl
Train records: 25312 -> /data/dsq/ScanNet/qa_jsonl/train.jsonl
Val/Test records: 1800 -> /data/dsq/ScanNet/qa_jsonl/val.jsonl
'''
#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys


EXCLUDED_TASKS = {"distance_infer_center_oc"}
ELIGIBLE_DIRS = ("fill", "judge", "select")


def list_task_dirs(split_dir):
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def choose_jsonl_dirs(task_dir):
    fill_dir = os.path.join(task_dir, "fill")
    if os.path.isdir(fill_dir):
        return [fill_dir]

    chosen = []
    for name in ("judge", "select"):
        candidate = os.path.join(task_dir, name)
        if os.path.isdir(candidate):
            chosen.append(candidate)
    return chosen


def collect_jsonl_paths(root_dir, split):
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"Missing split dir: {split_dir}")

    selected = []
    for task in list_task_dirs(split_dir):
        if task in EXCLUDED_TASKS:
            continue

        task_dir = os.path.join(split_dir, task)
        chosen_dirs = choose_jsonl_dirs(task_dir)
        if not chosen_dirs:
            print(f"Warning: no eligible dirs under {task_dir}", file=sys.stderr)
            continue

        for chosen_dir in chosen_dirs:
            jsonl_files = sorted(
                f for f in os.listdir(chosen_dir) if f.endswith(".jsonl")
            )
            if len(jsonl_files) != 1:
                raise RuntimeError(
                    f"Expected 1 jsonl in {chosen_dir}, found {len(jsonl_files)}: {jsonl_files}"
                )
            selected.append((task, os.path.join(chosen_dir, jsonl_files[0])))

    selected.sort(key=lambda item: item[1])
    return selected


def load_records(paths):
    records = []
    for task, path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {path}:{line_num}: {exc}"
                    ) from exc
                records.append(obj)
    return records


def split_train_test(records, per_type, seed):
    rng = random.Random(seed)
    type_to_indices = {}
    for idx, rec in enumerate(records):
        rec_type = rec.get("type")
        type_to_indices.setdefault(rec_type, []).append(idx)

    test_indices = set()
    sample_counts = {}
    for rec_type, indices in sorted(type_to_indices.items(), key=lambda x: str(x[0])):
        sample_size = min(per_type, len(indices))
        sample_counts[rec_type] = sample_size
        if sample_size:
            test_indices.update(rng.sample(indices, sample_size))

    train_records = [rec for i, rec in enumerate(records) if i not in test_indices]
    test_records = [rec for i, rec in enumerate(records) if i in test_indices]
    return train_records, test_records, type_to_indices, sample_counts


def sort_by_id(records):
    return sorted(records, key=lambda r: (r.get("id", ""), r.get("type", "")))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge ScanNet qa_jsonl and create all/train/val jsonl files."
    )
    parser.add_argument(
        "--root",
        default="/data/dsq/ScanNet/qa_jsonl",
        help="Root directory containing train/val subdirs.",
    )
    parser.add_argument(
        "--per-type-test",
        type=int,
        default=200,
        help="Number of samples per type for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--all-out",
        default="all.jsonl",
        help="Output filename for all data (written under root).",
    )
    parser.add_argument(
        "--train-out",
        default="train.jsonl",
        help="Output filename for train data (written under root).",
    )
    parser.add_argument(
        "--val-out",
        default="val.jsonl",
        help="Output filename for validation/test data (written under root).",
    )
    args = parser.parse_args()

    selected_paths = []
    for split in ("train", "val"):
        selected_paths.extend(collect_jsonl_paths(args.root, split))

    if not selected_paths:
        raise RuntimeError("No jsonl files selected. Check the input directories.")

    records = load_records(selected_paths)
    train_records, test_records, type_to_indices, sample_counts = split_train_test(
        records, args.per_type_test, args.seed
    )

    all_sorted = sort_by_id(records)
    train_sorted = sort_by_id(train_records)
    test_sorted = sort_by_id(test_records)

    all_path = os.path.join(args.root, args.all_out)
    train_path = os.path.join(args.root, args.train_out)
    val_path = os.path.join(args.root, args.val_out)

    write_jsonl(all_path, all_sorted)
    write_jsonl(train_path, train_sorted)
    write_jsonl(val_path, test_sorted)

    print(f"Selected jsonl files: {len(selected_paths)}")
    for rec_type in sorted(type_to_indices.keys(), key=lambda x: str(x)):
        total = len(type_to_indices[rec_type])
        sampled = sample_counts.get(rec_type, 0)
        print(f"type={rec_type} total={total} test_sampled={sampled}")

    print(f"All records: {len(all_sorted)} -> {all_path}")
    print(f"Train records: {len(train_sorted)} -> {train_path}")
    print(f"Val/Test records: {len(test_sorted)} -> {val_path}")


if __name__ == "__main__":
    main()
