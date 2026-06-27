'''
遍历一个目录下的所有 .jsonl 文件，对每个文件的每一行 JSON 样本做如下事情：

1）读取样本的 image 字段，取出其中的图片路径 src_path（优先取列表的第 0 张）。
2）从样本 id 解析出 scene_dir（场景目录名）。
3）把 src_path 指向的真实图片文件复制到一个新的数据根目录 new_root 下的固定结构路径中：
new_root/images/color/<scene_dir>/<原图片文件名>
4）如果复制成功，则把样本的 image 字段更新为 [dst_path]，其中 dst_path 是新位置的绝对路径；样本被保留。
5）如果样本缺 image 或图片复制失败，则丢弃该样本。
6）该 jsonl 文件中所有保留样本会按 scene_dir 排序后写回原文件（除非 dry-run），可选写回前生成 .bak 备份。

处理文件数: 52
处理样本行数: 66026
保留样本: 66026
删除样本: 0
实际修改的文件数: 52
'''
import os
import json
import argparse
import shutil

def iter_jsonl_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".jsonl"):
                yield os.path.join(dirpath, name)

def extract_scene_dir(sample_id):
    if not isinstance(sample_id, str):
        return None
    if "_" not in sample_id:
        return None
    return sample_id.rsplit("_", 1)[0]

def copy_image_to_new_root(src_path, new_root, scene_dir):
    if not src_path or not os.path.isfile(src_path):
        return None

    base_name = os.path.basename(src_path)
    if not scene_dir:
        return None

    dst_dir = os.path.join(new_root, "images", "color", scene_dir)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, base_name)
    if not os.path.isfile(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path

def process_file(path, new_root, dry_run=False, backup=False):
    tmp_path = path + ".tmp"
    changed = False
    total = 0
    kept = 0
    dropped = 0
    records = []

    with open(path, "r", encoding="utf-8") as f_in:
        for line_no, line in enumerate(f_in, 1):
            line = line.rstrip("\n")
            if line == "":
                continue

            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON解析失败: {path} 第{line_no}行: {e}") from e

            image_list = obj.get("image")
            if isinstance(image_list, list) and image_list:
                src_path = image_list[0]
            elif isinstance(image_list, str):
                src_path = image_list
            else:
                # 没有image字段，直接丢弃
                dropped += 1
                changed = True
                continue

            scene_dir = extract_scene_dir(obj.get("id"))
            dst_path = copy_image_to_new_root(src_path, new_root, scene_dir)
            if not dst_path:
                dropped += 1
                changed = True
                continue

            obj["image"] = [dst_path]
            records.append(obj)
            kept += 1
            changed = True

    # 按 scene_dir 排序
    records.sort(key=lambda x: (extract_scene_dir(x.get("id")) or ""))

    if dry_run:
        return total, kept, dropped, changed

    with open(tmp_path, "w", encoding="utf-8") as f_out:
        for obj in records:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if backup:
        shutil.copy2(path, path + ".bak")
    os.replace(tmp_path, path)

    return total, kept, dropped, changed

def main():
    parser = argparse.ArgumentParser(description="Copy images and update jsonl image field, then sort by scene_dir.")
    parser.add_argument("--jsonl-root", default="/data/dsq/ScanNet/qa_jsonl", help="jsonl根目录")
    parser.add_argument("--new-root", default="/data/dsq/ScanNet/qa_jsonl", help="新数据根目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写回")
    parser.add_argument("--backup", action="store_true", help="写回前备份为 .bak")
    args = parser.parse_args()

    total_files = 0
    total_lines = 0
    total_kept = 0
    total_dropped = 0
    total_changed_files = 0

    for fp in iter_jsonl_files(args.jsonl_root):
        total_files += 1
        lines, kept, dropped, changed = process_file(fp, args.new_root, args.dry_run, args.backup)
        total_lines += lines
        total_kept += kept
        total_dropped += dropped
        if changed:
            total_changed_files += 1

    print(f"处理文件数: {total_files}")
    print(f"处理样本行数: {total_lines}")
    print(f"保留样本: {total_kept}")
    print(f"删除样本: {total_dropped}")
    if args.dry_run:
        print("dry-run模式：未写回文件")
    else:
        print(f"实际修改的文件数: {total_changed_files}")

if __name__ == "__main__":
    main()