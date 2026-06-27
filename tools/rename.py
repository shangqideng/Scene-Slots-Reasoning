import json
import os
import re

INPUT_PATH = "/data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename.jsonl"
SAM2_PATH = "/data/dsq/ScanNet/qa_jsonl/all.sam2.scene_slot.jsonl"
OUTPUT_PATH = "/data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.jsonl"

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_floats(text, n=None):
    if text is None:
        return None
    nums = FLOAT_RE.findall(str(text))
    if n is not None and len(nums) < n:
        return None
    return [float(x) for x in nums[:n]] if n else [float(x) for x in nums]

def parse_center(center):
    if isinstance(center, (list, tuple)) and len(center) == 3:
        return [float(x) for x in center]
    if isinstance(center, str):
        vals = _parse_floats(center, 3)
        if vals is not None:
            return vals
    return None

def parse_bbox(bbox):
    # already list/tuple of 6?
    if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
        return [float(x) for x in bbox]

    # dict with "x_min ~ x_max" style
    if isinstance(bbox, dict):
        xr = bbox.get("x_min ~ x_max") or bbox.get("x_min~x_max")
        yr = bbox.get("y_min ~ y_max") or bbox.get("y_min~y_max")
        zr = bbox.get("z_min ~ z_max") or bbox.get("z_min~z_max")

        xr = _parse_floats(xr, 2) if xr is not None else None
        yr = _parse_floats(yr, 2) if yr is not None else None
        zr = _parse_floats(zr, 2) if zr is not None else None

        if xr and yr and zr:
            return [xr[0], yr[0], zr[0], xr[1], yr[1], zr[1]]

    return None

def _norm_text(text):
    if text is None:
        return None
    return str(text).strip().lower()


def build_sam2_index(path):
    index = {}
    total_2d = 0
    total_objects = 0
    if not os.path.isfile(path):
        raise RuntimeError(f"sam2 jsonl not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = sample.get("id")
            scene_slot = sample.get("scene_slot")
            objects = []
            if isinstance(scene_slot, dict):
                objects = scene_slot.get("objects") or []
            elif isinstance(scene_slot, list):
                objects = scene_slot
            if not sid or not isinstance(objects, list):
                continue
            obj_list = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                total_objects += 1
                name = _norm_text(obj.get("name"))
                color_ann = _norm_text(obj.get("color_annotation"))
                bbox2d = obj.get("2dmask_bbox")
                obj_list.append(
                    {
                        "name": name,
                        "color_annotation": color_ann,
                        "bbox2d": bbox2d,
                    }
                )
                if bbox2d is not None:
                    total_2d += 1
            if obj_list:
                index[sid] = obj_list
    return index, total_2d, total_objects


def convert_scene_slot(scene_slot, sam2_index=None, sample_id=None):
    if not isinstance(scene_slot, list):
        return scene_slot

    converted = []
    for obj in scene_slot:
        if not isinstance(obj, dict):
            converted.append(obj)
            continue

        name = obj.get("object name", obj.get("name"))
        color_ann = obj.get("color_annotation")
        center_3d = parse_center(obj.get("center", obj.get("center_3d")))
        bbox_3d = parse_bbox(obj.get("bounding box", obj.get("bbox_3d")))
        bbox_2d = None
        if sam2_index is not None and sample_id is not None and name is not None:
            sam2_list = sam2_index.get(sample_id)
            if sam2_list:
                target_name = _norm_text(name)
                target_color = _norm_text(color_ann)
                match_idx = None
                # 1) name + color_annotation
                if target_name and target_color:
                    for i, item in enumerate(sam2_list):
                        if item["name"] == target_name and item["color_annotation"] == target_color:
                            match_idx = i
                            break
                # 2) color_annotation only
                if match_idx is None and target_color:
                    for i, item in enumerate(sam2_list):
                        if item["color_annotation"] == target_color:
                            match_idx = i
                            break
                # 3) name only
                if match_idx is None and target_name:
                    for i, item in enumerate(sam2_list):
                        if item["name"] == target_name:
                            match_idx = i
                            break
                if match_idx is not None:
                    bbox_2d = sam2_list.pop(match_idx).get("bbox2d")

        new_obj = {}
        if name is not None:
            new_obj["name"] = name
        if color_ann is not None:
            new_obj["color_annotation"] = color_ann
        new_obj["2dmask_bbox"] = bbox_2d
        if center_3d is not None:
            new_obj["center_3d"] = center_3d
        if bbox_3d is not None:
            new_obj["bbox_3d"] = bbox_3d
        if "ave_depth" in obj:
            new_obj["ave_depth"] = obj.get("ave_depth")

        # 保留其他字段（但去掉旧字段）
        drop_keys = {
            "id",
            "object name",
            "name",
            "center",
            "center_3d",
            "bounding box",
            "bbox_3d",
            "color_annotation",
            "ave_depth",
        }
        for k, v in obj.items():
            if k in drop_keys:
                continue
            new_obj[k] = v

        converted.append(new_obj)

    return converted

def main():
    if not os.path.isfile(INPUT_PATH):
        raise RuntimeError(f"input not found: {INPUT_PATH}")

    sam2_index, sam2_2d_count, sam2_obj_count = build_sam2_index(SAM2_PATH)
    total = 0
    invalid = 0
    added_2d = 0
    renamed_obj_count = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for line_idx, line in enumerate(f_in):
            # Preserve line order: write exactly one output line per input line.
            if not line.strip():
                f_out.write("\n")
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                f_out.write(line)
                continue

            total += 1
            scene_slot = convert_scene_slot(
                sample.get("scene_slot"),
                sam2_index=sam2_index,
                sample_id=sample.get("id"),
            )
            if isinstance(scene_slot, list):
                for obj in scene_slot:
                    if isinstance(obj, dict) and obj.get("2dmask_bbox") is not None:
                        added_2d += 1
                    if isinstance(obj, dict):
                        renamed_obj_count += 1
            sample["scene_slot"] = scene_slot
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Total samples: {total}")
    print(f"Invalid lines kept: {invalid}")
    print(f"Objects in sam2 jsonl: {sam2_obj_count}")
    print(f"Objects in output jsonl: {renamed_obj_count}")
    print(f"2dmask_bbox in sam2 jsonl: {sam2_2d_count}")
    print(f"2dmask_bbox copied into output: {added_2d}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()