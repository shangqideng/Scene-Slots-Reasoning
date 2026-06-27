'''
读取一个或多个 .jsonl 标注文件（每行一个 JSON 样本）。
对每个样本，根据 JSON 里带有的“点标注字段”和“框标注字段”，在对应的原始图像上画出圆点与矩形框。
把“画好标注的图”保存到 out_root/images/color_annotated/<scene_dir>/<id>.jpg。
将该样本的 image 字段更新为新生成的标注图路径，并把整个 .jsonl 写回原文件（可选备份）

全量处理完成
处理jsonl文件数: 52
生成图片数: 66026
失败样本数: 0
'''
import os
import json
import argparse
from PIL import Image, ImageDraw

# 颜色映射
COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

# 遍历一个目录下的所有 .jsonl 文件
def iter_jsonl_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith(".jsonl"):
                yield os.path.join(dirpath, name)

# 从样本 id 解析出 scene_dir（场景目录名）。
def extract_scene_dir(sample_id):
    if not isinstance(sample_id, str):
        return None
    if "_" not in sample_id:
        return None
    return sample_id.rsplit("_", 1)[0]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

###########################################################
def scale_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return (
        x1 / 1000.0 * width,
        y1 / 1000.0 * height,
        x2 / 1000.0 * width,
        y2 / 1000.0 * height,
    )

def scale_point(point, width, height):
    x, y = point
    return (x / 1000.0 * width, y / 1000.0 * height)

# 画圆点（按官方方式：填充圆）
def draw_points(draw, image, points, color, radius=16):
    for (x, y) in points:
        sx, sy = scale_point((x, y), image.width, image.height)
        draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=color)

# 画矩形框（按官方方式：加粗、外扩）
def draw_bboxes(draw, image, bboxes, color, stroke=16):
    for (x1, y1, x2, y2) in bboxes:
        sx1, sy1, sx2, sy2 = scale_bbox((x1, y1, x2, y2), image.width, image.height)
        extend = stroke * 7 / 8
        draw.rectangle(
            [sx1 - extend, sy1 - extend, sx2 + extend, sy2 + extend],
            outline=color,
            width=stroke,
        )
###########################################################

# 解析标注字段 （解析出圆点与矩形框）
def parse_annot_fields(obj):
    points = []
    bboxes = []

    for key, val in obj.items(): # 遍历样本中的所有字段
        if not isinstance(key, str) or not isinstance(val, list):
            continue
        if key.endswith("_point"): # 如果字段名以 "_point" 结尾
            color_name = key.replace("_point", "") # 去掉 "_point" 后缀
            if color_name in COLOR_MAP:
                pts = [tuple(p) for p in val if isinstance(p, list) and len(p) == 2] 
                if pts:
                    points.append((COLOR_MAP[color_name], pts)) # 将圆点添加到 points 列表中
        if key.endswith("_bbox"): # 如果字段名以 "_bbox" 结尾
            color_name = key.replace("_bbox", "")
            if color_name in COLOR_MAP:
                boxes = [tuple(b) for b in val if isinstance(b, list) and len(b) == 4]
                if boxes:
                    bboxes.append((COLOR_MAP[color_name], boxes)) # 将矩形框添加到 bboxes 列表中
    return points, bboxes

# 生成标注图
def make_annotated_image(src_path, out_path, obj):
    if not os.path.isfile(src_path):
        return False

    img = Image.open(src_path).convert("RGB") # 打开原始图像并转换为RGB模式
    draw = ImageDraw.Draw(img) # 创建一个绘图对象

    points, bboxes = parse_annot_fields(obj)
    for color, pts in points:
        draw_points(draw, img, pts, color)
    for color, boxes in bboxes:
        draw_bboxes(draw, img, boxes, color)

    ensure_dir(os.path.dirname(out_path)) # 确保输出目录存在
    img.save(out_path, quality=95)
    return True # 返回True表示生成标注图成功

# 处理单个样本(只处理 id == target_id 的那一条)
def process_single(jsonl_file, target_id, out_root):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    new_records = []
    hit = False

    for line in lines:
        obj = json.loads(line)
        if obj.get("id") != target_id:
            new_records.append(obj)
            continue

        image_list = obj.get("image", [])
        if isinstance(image_list, list) and image_list:
            src_path = image_list[0]
        elif isinstance(image_list, str):
            src_path = image_list
        else:
            raise RuntimeError("该样本无image字段")

        scene_dir = extract_scene_dir(obj.get("id"))
        out_dir = os.path.join(out_root, "images", "color_annotated", scene_dir)
        out_path = os.path.join(out_dir, f"{obj['id']}.jpg")

        ok = make_annotated_image(src_path, out_path, obj)
        if not ok:
            raise RuntimeError("生成标注图失败")

        obj["image"] = [out_path]
        new_records.append(obj)
        hit = True

    if not hit:
        raise RuntimeError(f"未找到id={target_id}的样本")
    return new_records

# 处理所有样本(处理整个文件)
def process_all(jsonl_file, out_root):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    new_records = []
    generated = 0
    failed = 0
    for line in lines:
        obj = json.loads(line)
        image_list = obj.get("image", [])
        if isinstance(image_list, list) and image_list:
            src_path = image_list[0]
        elif isinstance(image_list, str):
            src_path = image_list
        else:
            failed += 1
            continue

        scene_dir = extract_scene_dir(obj.get("id"))
        out_dir = os.path.join(out_root, "images", "color_annotated", scene_dir)
        out_path = os.path.join(out_dir, f"{obj['id']}.jpg")

        ok = make_annotated_image(src_path, out_path, obj)
        if not ok:
            failed += 1
            continue

        obj["image"] = [out_path]
        new_records.append(obj)
        generated += 1

    return new_records, generated, failed

def write_jsonl(path, records, backup=False):
    if backup:
        import shutil
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 单样本：
# --single-file <path/to/file.jsonl> --single-id <id>
# 全量：
# --all（同时会递归遍历 --jsonl-root 下所有 jsonl）
def main():
    parser = argparse.ArgumentParser(description="Annotate images and update jsonl.")
    parser.add_argument("--jsonl-root", default="/data/dsq/ScanNet/qa_jsonl", help="jsonl根目录")
    parser.add_argument("--out-root", default="/data/dsq/ScanNet/qa_jsonl", help="输出根目录")
    parser.add_argument("--single-file", help="单文件模式：指定jsonl文件")
    parser.add_argument("--single-id", help="单样本模式：指定样本id")
    parser.add_argument("--all", action="store_true", help="批处理所有jsonl")
    parser.add_argument("--backup", action="store_true", help="写回前备份为 .bak")
    args = parser.parse_args()

    if args.single_file and args.single_id:
        records = process_single(args.single_file, args.single_id, args.out_root)
        print("单样本处理完成")
        return

    if args.all:
        total_files = 0
        total_generated = 0
        total_failed = 0
        for fp in iter_jsonl_files(args.jsonl_root):
            records, generated, failed = process_all(fp, args.out_root)
            write_jsonl(fp, records, args.backup)
            total_files += 1
            total_generated += generated
            total_failed += failed
        print("全量处理完成")
        print(f"处理jsonl文件数: {total_files}")
        print(f"生成图片数: {total_generated}")
        print(f"失败样本数: {total_failed}")
        return

    raise RuntimeError("请指定 --single-file + --single-id 或 --all")

if __name__ == "__main__":
    main()