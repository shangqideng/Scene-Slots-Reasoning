"""
清理jsonl文件，删除没有rationale字段的样本行。

使用方法：
python RL/clean_incomplete_jsonl.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.cleaned.jsonl

python RL/clean_incomplete_jsonl.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.cleaned.jsonl
"""
import argparse
import json
import os


def clean_jsonl(input_path, output_path):
    """删除没有rationale字段的样本行"""
    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file not found: {input_path}")
    
    total_lines = 0
    kept_lines = 0
    removed_lines = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            total_lines += 1
            
            if not line.strip():
                # Keep empty lines
                f_out.write(line)
                continue
            
            try:
                obj = json.loads(line)
                # Check if rationale field exists and is not empty
                if "rationale" in obj and obj["rationale"]:
                    # Keep this line
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept_lines += 1
                else:
                    # Remove this line (no rationale field or empty)
                    removed_lines += 1
            except json.JSONDecodeError:
                # Keep malformed JSON lines (preserve original)
                f_out.write(line)
                kept_lines += 1
    
    print(f"Cleaning completed:")
    print(f"  Total lines: {total_lines}")
    print(f"  Kept lines: {kept_lines}")
    print(f"  Removed lines: {removed_lines}")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean jsonl file by removing lines without rationale field")
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Input jsonl file path"
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output cleaned jsonl file path"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    clean_jsonl(args.input_jsonl, args.output_jsonl)


if __name__ == "__main__":
    main()
