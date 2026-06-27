'''
它的核心工作是：遍历一个目录下的所有 .jsonl 文件，对其中每一行 JSON 样本：
从样本的 id 中解析出 scene_dir（场景目录名）。
结合一个“真实图片根目录” raw_root，去磁盘上验证图片文件是否真的存在。
如果找到真实图片：
把样本的 id 统一重命名为全局唯一（保持 scene 前缀不变，后缀按处理顺序递增）。
把样本的 image 字段统一改成 [真实绝对路径] 这种列表形式（只保留匹配到的那一张）。
在同一路径下根据图片名生成点云路径 xxx_full_fov_camera.ply，写入 ply_path 字段（列表，只有1个元素）。
把 depth 字段置空为 []。
如果找不到真实图片：
整个删除该行
最终把修改后的样本写回原 .jsonl 文件（可选 dry-run，不写回；可选备份 .bak）。


处理文件数: 52
处理样本行数: 3183279
保留样本: 66026(6.6w个QA对)
删除样本: 3117253
实际修改的文件数: 52
'''
import os
import json
import argparse
import shutil

# 遍历一个目录下的所有 .jsonl 文件
def iter_jsonl_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".jsonl"):
                yield os.path.join(dirpath, name)

# 从样本的 id 中解析出 scene_dir（场景目录名）。
def extract_scene_dir(sample_id):
    if not isinstance(sample_id, str):
        return None
    if "_" not in sample_id:
        return None
    return sample_id.rsplit("_", 1)[0]

# 从样本 id 中提取前缀（用于全局唯一重命名）
def extract_id_prefix(sample_id):
    if not isinstance(sample_id, str):
        return None
    if "_" not in sample_id:
        return sample_id if sample_id else None
    return sample_id.rsplit("_", 1)[0]

# 构造全局唯一的 id：保持前缀不变，后缀按处理顺序递增
def build_unique_id(sample_id, id_counters):
    prefix = extract_id_prefix(sample_id)
    if not prefix:
        prefix = "unknown"
    next_idx = id_counters.get(prefix, 0)
    id_counters[prefix] = next_idx + 1
    return f"{prefix}_{next_idx}"

# 只取最后的文件名 xxx.jpg，然后去 raw_root/scene_dir/xxx.jpg 这个位置找
def resolve_image_path(raw_root, scene_dir, image_field):
    if not scene_dir:
        return None

    candidates = []
    if isinstance(image_field, list):
        candidates = [x for x in image_field if isinstance(x, str)] # 把列表中的所有字符串元素提取出来
    elif isinstance(image_field, str):
        candidates = [image_field] # 如果是字符串，直接添加到列表中

    for v in candidates:
        image_name = os.path.basename(v) # 只取最后的文件名 xxx.jpg
        if not image_name:
            continue
        candidate = os.path.join(raw_root, scene_dir, image_name) # 拼接成 raw_root/scene_dir/xxx.jpg 这个位置找
        if os.path.isfile(candidate):
            return candidate # 如果找到真实图片，返回真实绝对路径
    return None

# 根据图片路径生成对应点云路径：xxx.jpg -> xxx_full_fov_camera.ply
def build_ply_path(image_path):
    if not image_path:
        return None
    base_name = os.path.basename(image_path)
    stem, _ = os.path.splitext(base_name)
    if not stem:
        return None
    return os.path.join(os.path.dirname(image_path), f"{stem}_full_fov_camera.ply")

# 处理一个 .jsonl 文件
def process_file(path, raw_root, id_counters, dry_run=False, backup=False):
    tmp_path = path + ".tmp"
    changed = False
    total = 0
    kept = 0
    dropped = 0

    with open(path, "r", encoding="utf-8") as f_in, open(tmp_path, "w", encoding="utf-8") as f_out:
        # 遍历文件中的每一行
        for line_no, line in enumerate(f_in, 1):
            line = line.rstrip("\n")
            if line == "":
                # 空行原样保留
                f_out.write("\n")
                continue

            total += 1
            try:
                obj = json.loads(line) # 解析成字典
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON解析失败: {path} 第{line_no}行: {e}") from e

            scene_dir = extract_scene_dir(obj.get("id")) # 从样本的 id 中解析出 scene_dir（场景目录名）。
            image_path = resolve_image_path(raw_root, scene_dir, obj.get("image")) # 结合一个“真实图片根目录” raw_root，去磁盘上验证图片文件是否真的存在。

            if not image_path:
                # 找不到图片，直接丢弃该样本
                dropped += 1
                changed = True
                continue

            # 找到图片：更新image为真实路径，添加ply_path，depth置空
            obj["id"] = build_unique_id(obj.get("id"), id_counters)
            obj["image"] = [image_path]
            ply_path = build_ply_path(image_path)
            if ply_path:
                obj["ply_path"] = [ply_path]
            obj["depth"] = []
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    if dry_run:
        os.remove(tmp_path)
    else:
        if backup:
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)

    return total, kept, dropped, changed

def main():
    parser = argparse.ArgumentParser(description="Update jsonl image/depth fields based on real data paths.")
    parser.add_argument("--jsonl-root", default="/data/dsq/ScanNet/qa_jsonl", help="jsonl根目录")
    parser.add_argument("--raw-root", default="/data/dsq/ScanNet/raw_select", help="真实图片根目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写回")
    parser.add_argument("--backup", action="store_true", help="写回前备份为 .bak")
    args = parser.parse_args()

    total_files = 0
    total_lines = 0
    total_kept = 0
    total_dropped = 0
    total_changed_files = 0

    # 遍历所有 .jsonl 文件
    id_counters = {}
    for fp in iter_jsonl_files(args.jsonl_root):
        total_files += 1
        lines, kept, dropped, changed = process_file(
            fp, args.raw_root, id_counters, args.dry_run, args.backup
        ) # 处理一个 .jsonl 文件
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