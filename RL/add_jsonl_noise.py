import argparse
import json
import random
from pathlib import Path


DEFAULT_INPUT_PATH = "/data/dsq/ScanNet/qa_jsonl/val.question_scene_slot_correct_rename_add2dmask.jsonl"


def add_noise_to_center(center_3d, noise_range=0.2):
    """For each dimension in center_3d, add uniform noise in [-noise_range, noise_range]."""
    if not isinstance(center_3d, list):
        return center_3d

    noisy_center = []
    for value in center_3d:
        if isinstance(value, (int, float)):
            noise = random.uniform(-noise_range, noise_range)
            noisy_center.append(round(value + noise, 2))
        else:
            noisy_center.append(value)
    return noisy_center


def process_jsonl(input_path, output_path, noise_range=0.2):
    total_lines = 0
    modified_objects = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            sample = json.loads(line)
            scene_slot = sample.get("scene_slot")

            if isinstance(scene_slot, list):
                for obj in scene_slot:
                    if not isinstance(obj, dict):
                        continue
                    if "center_3d" in obj:
                        obj["center_3d"] = add_noise_to_center(
                            obj["center_3d"], noise_range=noise_range
                        )
                        modified_objects += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return total_lines, modified_objects


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add uniform random noise in [-noise, noise] to center_3d in scene_slot."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Input jsonl path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output jsonl path. Default: input filename with _addnoise suffix.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.2,
        help="Noise magnitude for each dimension in center_3d.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_addnoise{input_path.suffix}")
    else:
        output_path = Path(args.output)

    total_lines, modified_objects = process_jsonl(
        input_path=str(input_path),
        output_path=str(output_path),
        noise_range=args.noise,
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Processed samples: {total_lines}")
    print(f"Modified objects with center_3d: {modified_objects}")
    print("Done. Original jsonl is untouched.")


if __name__ == "__main__":
    main()
