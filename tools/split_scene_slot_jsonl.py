"""
根据 train/val 的 id 列表，将 all.sam2.scene_slot.jsonl 划分为 train/val。

用法示例：
python /home/lxp/Ground_reasoning/tools/split_scene_slot_jsonl.py \
  --src /data/dsq/ScanNet/qa_jsonl/all.qwen_scene_slot.max10.jsonl \
  --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.jsonl \
  --train-out /data/dsq/ScanNet/qa_jsonl/train.qwen_scene_slot.jsonl \
  --val-out /data/dsq/ScanNet/qa_jsonl/val.qwen_scene_slot.jsonl \
  --drop-empty \
  --clean-out /data/dsq/ScanNet/qa_jsonl/all.qwen_scene_slot.nonempty.jsonl
对qwen_scene_slot:
splitting: 27112lines [00:00, 36262.37lines/s]
Split done: total=27112, train=23987, val=1709, unknown=0, drop_empty=1416

python /home/lxp/Ground_reasoning/tools/split_scene_slot_jsonl.py \
  --src /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.jsonl \
  --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.jsonl \
  --train-out /data/dsq/ScanNet/qa_jsonl/train.question_scene_slot_correct_rename_add2dmask.jsonl \
  --val-out /data/dsq/ScanNet/qa_jsonl/val.question_scene_slot_correct_rename_add2dmask.jsonl \
  --drop-empty \
  --clean-out /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.nonempty.jsonl
对question_scene_slot:
splitting: 22246lines [00:00, 52133.58lines/s]
Split done: total=22246, train=20277, val=1415, unknown=0, drop_empty=554


python /home/lxp/Ground_reasoning/tools/split_scene_slot_jsonl.py \
  --src /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.jsonl \
  --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.jsonl \
  --train-out /data/dsq/ScanNet/qa_jsonl/train.question.rationale.modified_ds.jsonl \
  --val-out /data/dsq/ScanNet/qa_jsonl/val.question.rationale.modified_ds.jsonl \
  --drop-empty \
  --clean-out /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.nonempty.jsonl
splitting: 27112lines [00:00, 41328.72lines/s]
Split done: total=27112, train=25312, val=1800, unknown=0, drop_empty=0
"""
import argparse
import json
import os

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def load_ids(jsonl_path):
    ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if sid is not None:
                ids.add(sid)
    return ids


def is_scene_slot_empty(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return True
    if "scene_slot" not in sample:
        return True
    slot = sample.get("scene_slot")
    if slot is None:
        return True
    if isinstance(slot, str):
        return not slot.strip()
    if isinstance(slot, (list, dict, tuple, set)):
        return len(slot) == 0
    return False


def split_scene_slot(
    src,
    train_ids,
    val_ids,
    train_out,
    val_out,
    keep_unknown=False,
    drop_empty=False,
    clean_out=None,
):
    total = 0
    train_cnt = 0
    val_cnt = 0
    unknown_cnt = 0
    empty_cnt = 0
    progress = tqdm(desc="splitting", unit="lines") if tqdm else None
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(val_out), exist_ok=True)
    clean_fp = None
    if clean_out:
        clean_dir = os.path.dirname(clean_out)
        if clean_dir:
            os.makedirs(clean_dir, exist_ok=True)
        clean_fp = open(clean_out, "w", encoding="utf-8")
    with open(src, "r", encoding="utf-8") as f_in, open(
        train_out, "w", encoding="utf-8"
    ) as f_train, open(val_out, "w", encoding="utf-8") as f_val:
        for line in f_in:
            if not line.strip():
                if progress is not None:
                    progress.update(1)
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if progress is not None:
                    progress.update(1)
                continue
            if drop_empty and is_scene_slot_empty(obj):
                empty_cnt += 1
                if progress is not None:
                    progress.update(1)
                continue
            if clean_fp is not None:
                clean_fp.write(line)
            sid = obj.get("id")
            if sid in train_ids:
                f_train.write(line)
                train_cnt += 1
            elif sid in val_ids:
                f_val.write(line)
                val_cnt += 1
            else:
                unknown_cnt += 1
                if keep_unknown:
                    f_train.write(line)
                    train_cnt += 1
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    if clean_fp is not None:
        clean_fp.close()
    return total, train_cnt, val_cnt, unknown_cnt, empty_cnt


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Split scene_slot jsonl by id lists.")
    parser.add_argument("--src", required=True, help="all.sam2.scene_slot.jsonl")
    parser.add_argument("--train-jsonl", required=True, help="train.sam2.jsonl")
    parser.add_argument("--val-jsonl", required=True, help="val.sam2.jsonl")
    parser.add_argument("--train-out", required=True, help="train.sam2.scene_slot.jsonl")
    parser.add_argument("--val-out", required=True, help="val.sam2.scene_slot.jsonl")
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Write ids not in train/val into train output",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop samples with empty/missing scene_slot.",
    )
    parser.add_argument(
        "--clean-out",
        default=None,
        help="Optional output jsonl containing only non-empty scene_slot samples.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not os.path.isfile(args.src):
        raise RuntimeError(f"src not found: {args.src}")
    if not os.path.isfile(args.train_jsonl):
        raise RuntimeError(f"train jsonl not found: {args.train_jsonl}")
    if not os.path.isfile(args.val_jsonl):
        raise RuntimeError(f"val jsonl not found: {args.val_jsonl}")

    train_ids = load_ids(args.train_jsonl)
    val_ids = load_ids(args.val_jsonl)
    total, train_cnt, val_cnt, unknown_cnt, empty_cnt = split_scene_slot(
        args.src,
        train_ids,
        val_ids,
        args.train_out,
        args.val_out,
        keep_unknown=args.keep_unknown,
        drop_empty=args.drop_empty,
        clean_out=args.clean_out,
    )
    print(
        "Split done: total="
        f"{total}, train={train_cnt}, val={val_cnt}, unknown={unknown_cnt}, "
        f"drop_empty={empty_cnt}"
    )
    if args.clean_out:
        print(f"Clean out: {args.clean_out}")
    print(f"Train out: {args.train_out}")
    print(f"Val out: {args.val_out}")


if __name__ == "__main__":
    main()
