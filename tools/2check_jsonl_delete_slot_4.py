'''
Total parsed samples: 27112
Removed (scene_slot > 4): 4866
Kept: 22246
Invalid JSON lines kept: 0
Output: /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot.max4.jsonl

Total parsed samples: 27112
Removed (scene_slot > 10): 3815
Kept: 23297
Invalid JSON lines kept: 0
Output: /data/dsq/ScanNet/qa_jsonl/all.qwen_scene_slot.max10.jsonl

'''
import json
import os

INPUT_PATH = "/data/dsq/ScanNet/qa_jsonl/all.question_scene_slot.jsonl"
OUTPUT_PATH = "/data/dsq/ScanNet/qa_jsonl/all.question_scene_slot.max4.jsonl"

def is_too_many_slots(sample, max_slots=4):
    slots = sample.get("scene_slot")
    if isinstance(slots, list):
        return len(slots) > max_slots
    return False

def main():
    if not os.path.isfile(INPUT_PATH):
        raise RuntimeError(f"input not found: {INPUT_PATH}")

    total = 0
    removed = 0
    invalid = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # 保留无法解析的行，但计入 invalid
                invalid += 1
                f_out.write(line)
                continue

            total += 1
            if is_too_many_slots(sample, max_slots=4):
                removed += 1
                continue
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    kept = total - removed
    print(f"Total parsed samples: {total}")
    print(f"Removed (scene_slot > 4): {removed}")
    print(f"Kept: {kept}")
    print(f"Invalid JSON lines kept: {invalid}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()