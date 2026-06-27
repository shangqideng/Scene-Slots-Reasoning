"""
根据 sam2_seg 的 mask 结果，从点云投影筛选目标点集，计算 3D 属性并生成 scene_slot。

功能：
1) 单样本可视化：给定 jsonl + id，输出 scene_slot json 与叠加 mask 的图片（不改原文件）
python 6scene_slot_from_sam2.py \
  --jsonl /data/dsq/ScanNet/qa_jsonl/all.sam2.jsonl \
  --vis-id scene0001_00_2

2) 批处理：遍历 jsonl，给每行添加 scene_slot 字段，写入新 out_jsonl
python 6scene_slot_from_sam2.py \
  --jsonl /data/dsq/ScanNet/qa_jsonl/all.sam2.jsonl \
  --out-jsonl /data/dsq/ScanNet/qa_jsonl/all.sam2.scene_slot.jsonl

依赖：
- plyfile（读取 .ply）
- numpy, pillow, torch (可选), tqdm(可选)
"""
# 1、图像+q+a；
# 2、图像+q+a+2d的scene_slot；
# 3、图像+q+a+3d的scene_slot（先用z-buffer前景模式）

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    from plyfile import PlyData
except Exception as exc:  # pragma: no cover
    PlyData = None
    _PLY_IMPORT_ERROR = exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

DEFAULT_MASK_ROOT = "/data/dsq/ScanNet/qa_jsonl/images/sam2_masks"

COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (139, 69, 19),
    "gray": (128, 128, 128),
}


def extract_scene_dir(sample):
    sample_id = sample.get("id")
    if isinstance(sample_id, str) and "_" in sample_id:
        return sample_id.rsplit("_", 1)[0]
    ply_list = sample.get("ply_path")
    if isinstance(ply_list, list) and ply_list:
        ply_path = ply_list[0]
        if isinstance(ply_path, str) and "/raw_select/" in ply_path:
            parts = ply_path.split("/raw_select/")
            if len(parts) > 1:
                rest = parts[1]
                return rest.split("/")[0]
    return "unknown"


def resolve_intrinsic_path(sample, raw_root="/data/dsq/ScanNet/raw_select"):
    scene_dir = extract_scene_dir(sample)
    return os.path.join(raw_root, scene_dir, "intrinsic_color.txt")


def resolve_ply_path(sample):
    ply_list = sample.get("ply_path")
    if isinstance(ply_list, list) and ply_list and isinstance(ply_list[0], str):
        return ply_list[0]
    return None


def resolve_image_path(sample):
    # 优先用原始图像（由 ply_path 推断）
    ply_path = resolve_ply_path(sample)
    if ply_path and ply_path.endswith("_full_fov_camera.ply"):
        raw_img = ply_path.replace("_full_fov_camera.ply", ".jpg")
        if os.path.isfile(raw_img):
            return raw_img
    image_list = sample.get("image", [])
    if image_list:
        return image_list[0]
    return None


def load_intrinsic(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise RuntimeError(f"invalid intrinsic file: {path}")
    mat = []
    for line in lines[:4]:
        mat.append([float(x) for x in line.split()])
    fx = mat[0][0]
    fy = mat[1][1]
    cx = mat[0][2]
    cy = mat[1][2]
    return fx, fy, cx, cy


def load_ply_points(ply_path):
    if PlyData is None:
        raise RuntimeError(f"plyfile import failed: {_PLY_IMPORT_ERROR}")
    ply = PlyData.read(ply_path)
    verts = ply["vertex"]
    x = np.asarray(verts["x"], dtype=np.float32)
    y = np.asarray(verts["y"], dtype=np.float32)
    z = np.asarray(verts["z"], dtype=np.float32)
    points = np.stack([x, y, z], axis=1)
    return points


def project_points(points, fx, fy, cx, cy):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    valid = z > 1e-6
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    u[valid] = fx * (x[valid] / z[valid]) + cx
    v[valid] = fy * (y[valid] / z[valid]) + cy
    return u, v, z, valid


def load_mask(mask_root, mask_path):
    abs_path = os.path.join(mask_root, mask_path)
    if not os.path.isfile(abs_path):
        return None
    img = Image.open(abs_path).convert("L")
    arr = np.array(img)
    return arr > 0


def build_zbuffer(points, u, v, valid, width, height):
    u_idx = np.rint(u).astype(np.int32)
    v_idx = np.rint(v).astype(np.int32)
    in_frame = (
        valid
        & (u_idx >= 0)
        & (u_idx < width)
        & (v_idx >= 0)
        & (v_idx < height)
    )
    if not np.any(in_frame):
        return None, None, None
    base_idx = np.nonzero(in_frame)[0]
    pix_idx = v_idx[in_frame] * width + u_idx[in_frame]
    z_in = points[in_frame, 2]
    order = np.lexsort((z_in, pix_idx))
    pix_sorted = pix_idx[order]
    base_sorted = base_idx[order]
    _, first_pos = np.unique(pix_sorted, return_index=True)
    sel_idx = base_sorted[first_pos]
    return sel_idx, u_idx[sel_idx], v_idx[sel_idx]


def compute_object_stats_from_zbuffer(points, zbuf_idx, u_pix, v_pix, mask_bool):
    if mask_bool is None or zbuf_idx is None:
        return None, None, None
    h, w = mask_bool.shape
    valid = (
        (u_pix >= 0)
        & (u_pix < w)
        & (v_pix >= 0)
        & (v_pix < h)
    )
    if not np.any(valid):
        return None, None, None
    sel = valid & mask_bool[v_pix, u_pix]
    if not np.any(sel):
        return None, None, None
    pts = points[zbuf_idx[sel]]
    center_3d = [float(v) for v in pts.mean(axis=0).tolist()]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    bbox_3d = [
        float(mins[0]),
        float(mins[1]),
        float(mins[2]),
        float(maxs[0]),
        float(maxs[1]),
        float(maxs[2]),
    ]
    ave_depth = float(pts[:, 2].mean())
    return center_3d, bbox_3d, ave_depth


def build_scene_slot(
    sample,
    raw_root="/data/dsq/ScanNet/raw_select",
    mask_root_override=None,
):
    sam2_seg = sample.get("sam2_seg", {})
    objects = sam2_seg.get("objects", [])
    if not objects:
        return {"objects": []}

    ply_path = resolve_ply_path(sample)
    if not ply_path or not os.path.isfile(ply_path):
        return {"objects": []}

    intrinsic_path = resolve_intrinsic_path(sample, raw_root=raw_root)
    if not os.path.isfile(intrinsic_path):
        return {"objects": []}

    fx, fy, cx, cy = load_intrinsic(intrinsic_path)
    points = load_ply_points(ply_path)

    # image size from sam2_seg (w,h)
    img_w, img_h = sam2_seg.get("image_size", [0, 0])
    if not img_w or not img_h:
        first_mask_path = None
        if objects:
            first_mask_path = objects[0].get("mask_path")
        if first_mask_path:
            mask = load_mask(mask_root_override or DEFAULT_MASK_ROOT, first_mask_path)
            if mask is not None:
                img_h, img_w = mask.shape
    if not img_w or not img_h:
        image_path = resolve_image_path(sample)
        if image_path and os.path.isfile(image_path):
            with Image.open(image_path) as img:
                img_w, img_h = img.size
    if not img_w or not img_h:
        return {"objects": []}

    u, v, z, valid = project_points(points, fx, fy, cx, cy)
    zbuf_idx, u_pix, v_pix = build_zbuffer(points, u, v, valid, img_w, img_h)

    mask_root = mask_root_override or DEFAULT_MASK_ROOT
    scene_objects = []
    for obj in objects:
        mask_path = obj.get("mask_path")
        mask_bool = load_mask(mask_root, mask_path) if mask_path else None
        center_3d, bbox_3d, ave_depth = compute_object_stats_from_zbuffer(
            points, zbuf_idx, u_pix, v_pix, mask_bool
        )
        scene_objects.append(
            {
                "name": obj.get("name"),
                "color_annotation": obj.get("prompt_key"),
                "2dmask_bbox": obj.get("mask_bbox"),
                "center_3d": center_3d,
                "bbox_3d": bbox_3d,
                "ave_depth": ave_depth,
            }
        )
    return {"objects": scene_objects}


def overlay_masks(image, objects, mask_root):
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for obj in objects:
        color_name = obj.get("color") or "white"
        color = COLOR_MAP.get(color_name, (255, 255, 255))
        mask_path = obj.get("mask_path")
        if not mask_path:
            continue
        abs_path = os.path.join(mask_root, mask_path)
        if not os.path.isfile(abs_path):
            continue
        mask = Image.open(abs_path).convert("L")
        color_img = Image.new("RGBA", image.size, color + (0,))
        color_img.putalpha(mask)
        overlay = Image.alpha_composite(overlay, color_img)
    return Image.alpha_composite(base, overlay).convert("RGB")


def find_sample_by_id(jsonl_path, sample_id):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") == sample_id:
                return obj
    return None


def process_single(jsonl_path, sample_id, raw_root, mask_root_override=None):
    sample = find_sample_by_id(jsonl_path, sample_id)
    if sample is None:
        raise RuntimeError(f"sample id not found: {sample_id}")
    scene_slot = build_scene_slot(sample, raw_root=raw_root, mask_root_override=mask_root_override)
    image_path = resolve_image_path(sample)
    if not image_path or not os.path.isfile(image_path):
        raise RuntimeError("image not found for visualization")
    sam2_seg = sample.get("sam2_seg", {})
    mask_root = mask_root_override or DEFAULT_MASK_ROOT
    image = Image.open(image_path).convert("RGB")
    vis = overlay_masks(image, sam2_seg.get("objects", []), mask_root)

    out_json = f"{sample_id}_scene_slot.json"
    out_img = f"{sample_id}_scene_slot.png"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scene_slot, f, ensure_ascii=False)
    vis.save(out_img)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_img}")


def process_jsonl(in_jsonl, out_jsonl, raw_root, mask_root_override=None):
    progress = tqdm(desc="processing", unit="lines") if tqdm else None
    with open(in_jsonl, "r", encoding="utf-8") as f_in, open(
        out_jsonl, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            if not line.strip():
                f_out.write("\n")
                if progress is not None:
                    progress.update(1)
                continue
            obj = json.loads(line)
            scene_slot = build_scene_slot(
                obj, raw_root=raw_root, mask_root_override=mask_root_override
            )
            obj["scene_slot"] = scene_slot
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build scene_slot from sam2 masks and point cloud."
    )
    parser.add_argument("--jsonl", required=True, help="Input jsonl path")
    parser.add_argument("--out-jsonl", help="Output jsonl with scene_slot")
    parser.add_argument("--vis-id", help="Single sample id for visualization")
    parser.add_argument(
        "--raw-root",
        default="/data/dsq/ScanNet/raw_select",
        help="Root path for raw_select scenes",
    )
    parser.add_argument(
        "--mask-root",
        default=DEFAULT_MASK_ROOT,
        help="Mask root path",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not os.path.isfile(args.jsonl):
        raise RuntimeError(f"jsonl not found: {args.jsonl}")

    if args.vis_id:
        process_single(args.jsonl, args.vis_id, args.raw_root, args.mask_root)
        return

    if not args.out_jsonl:
        raise RuntimeError("out-jsonl is required for batch mode")
    process_jsonl(args.jsonl, args.out_jsonl, args.raw_root, args.mask_root)
    print(f"Saved: {args.out_jsonl}")


if __name__ == "__main__":
    main()
