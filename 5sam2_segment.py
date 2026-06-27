"""
把 QA 数据集（jsonl）里“问题文本 + 颜色标注的 point/bbox”转换成 SAM2 的分割提示（prompt），对图像做实例分割，然后把分割结果（mask、bbox、面积、RLE 等元数据）回写到 jsonl；同时支持对单个样本输出可视化图
输入（它假设你的 jsonl 每行一个 sample）
每个 sample 至少可能包含：
    image: 一个列表，脚本只用第一张 image[0]，并假设它是本地可读的图像路径
    conversations: 用来取出问题文本（human 的那条）
    若干标注字段：例如 red_point, green_bbox 等，格式是二维列表（脚本只取第一个框/点）
        point 期望 [[x, y]]
        bbox 期望 [[x1, y1, x2, y2]]
        并且这些坐标被默认认为是 0~1000 的归一化坐标系（后面会解释）
输出（两种模式）  
    单样本可视化模式：--vis-id <id>
        找到 jsonl 中 id == vis_id 的样本
        跑 SAM2 分割
        生成叠加 mask 的图，保存到 --vis-out（默认 <id>_sam2.png）
        在终端打印每个对象的 score

(data_process) lxp@node01:~/Ground_reasoning$ python 5sam2_segment.py --jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl --vis-id scene0001_00_34 --vis-out ./s
cene0001_00_34_sam2.png
Saved visualization: ./scene0001_00_34_sam2.png
Objects segmented: 3
- picture (red point): score=0.9727
- armchair (green point): score=0.9648
- couch (blue point): score=0.9453
(data_process) lxp@node01:~/Ground_reasoning$ python 5sam2_segment.py --jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl --vis-id scene0001_00_17 --vis-out ./s
cene0001_00_17_sam2.png
Saved visualization: ./scene0001_00_17_sam2.png
Objects segmented: 3
- armchair (red point): score=0.9648
- picture (green point): score=0.9727
- couch (blue point): score=0.9453
(data_process) lxp@node01:~/Ground_reasoning$ python 5sam2_segment.py --jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl --vis-id scene0001_00_0 --vis-out ./scene0001_00_0_sam2.png
Saved visualization: ./scene0001_00_0_sam2.png
Objects segmented: 2
- picture (red point): score=0.9727
- armchair (blue point): score=0.9648

批处理写回 jsonl 模式：--out-jsonl <path>
    遍历输入 jsonl
    对每个有 image 的样本，提取 prompts -> SAM2 分割 -> 结果序列化
    在 sample 中新增字段：sam2_seg
    输出到新的 jsonl（不会原地修改输入文件）:out_jsonl 的每一行仍然是原 sample 的 JSON，只是多了一个 sam2_seg 字段

torchrun --nproc_per_node=8 5sam2_segment.py \
  --jsonl /data/dsq/ScanNet/qa_jsonl/all.jsonl \
  --out-jsonl /data/dsq/ScanNet/qa_jsonl/all.sam2.jsonl



增加按 id 拆分已生成的 all.sam2.jsonl 功能，不需要再跑分割。你只需要提供 all.sam2.jsonl、train.jsonl、val.jsonl 即可拆出 train.sam2.jsonl 和 val.sam2.jsonl
python 5sam2_segment.py \
  --split-src /data/dsq/ScanNet/qa_jsonl/all.sam2.jsonl \
  --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.jsonl \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.jsonl

splitting: 27112lines [00:00, 55974.70lines/s]
Split done: total=27112, train=25312, val=1800, unknown=0
Train out: /data/dsq/ScanNet/qa_jsonl/train.sam2.jsonl
Val out: /data/dsq/ScanNet/qa_jsonl/val.sam2.jsonl
"""
import argparse
import contextlib
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2
except Exception as exc:  # pragma: no cover - import error handling
    SAM2ImagePredictor = None
    build_sam2 = None
    _IMPORT_ERROR = exc

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

COLOR_ALIASES = {
    "grey": "gray",
}

COLOR_PATTERN = "|".join(sorted(set(COLOR_MAP.keys()) | set(COLOR_ALIASES.keys())))
MENTION_RE = re.compile(
    r"\(\s*(" + COLOR_PATTERN + r")\s+(point|bbox)\s*\)",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "using",
    "use",
    "used",
    "with",
    "from",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "between",
    "based",
    "calculate",
    "judge",
    "determine",
    "what",
    "which",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "known",
    "depth",
    "distance",
    "center",
    "centre",
    "centers",
    "centres",
    "objects",
    "object",
    "observer",
    "observers",
    "perspective",
    "viewpoint",
    "position",
    "points",
    "point",
    "bbox",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "white",
    "black",
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
    "farther",
    "closer",
    "far",
    "near",
    "reference",
    "refer",
    "provide",
    "answer",
    "respond",
    "submit",
    "please",
    "only",
    "one",
    "number",
    "meters",
    "meter",
    "cm",
    "centimeter",
    "centimeters",
    "length",
    "width",
    "height",
    "size",
    "difference",
    "absolute",
    "relative",
    "question",
    "image",
    "attached",
    "initial",
    "moves",
    "move",
    "moved",
    "faces",
    "face",
    "where",
    "does",
    "do",
    "did",
    "how",
    "would",
    "measure",
    "measures",
    "measured",
    "measuring",
    "who",
    "whom",
    "into",
    "toward",
    "towards",
    "relation",
    "relationship",
    "state",
    "options",
    "option",
    "select",
    "pick",
    "choose",
    "response",
    "reply",
}


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


def normalize_color(color):
    color = color.lower()
    return COLOR_ALIASES.get(color, color)


def clean_object_name(text):
    text = text.strip()
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(center|centre)\s+of\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(distance\s+to|distance\s+of|depth\s+of|depth\s+to|known\s+depth\s+of)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip(" ,.;:\n\t")
    tokens = text.split()
    while tokens and tokens[-1].lower() in STOPWORDS:
        tokens.pop()
    if tokens:
        return " ".join(tokens)
    return text or "object"


def extract_object_name_from_left_context(left_text):
    left_text = left_text.replace("’", "'")
    left_text = re.sub(r"['’]s\b", "", left_text)
    last_delim = max(left_text.rfind(d) for d in [".", "?", "!", ";", "\n"])
    if last_delim != -1:
        left_text = left_text[last_delim + 1 :]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_\/]*", left_text)
    if not tokens:
        return "object"
    collected = []
    for tok in reversed(tokens):
        lower_tok = tok.lower()
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            if collected:
                break
            continue
        if lower_tok in STOPWORDS:
            if collected:
                break
            continue
        collected.append(tok)
        if len(collected) >= 4:
            break
    if not collected:
        return clean_object_name(tokens[-1])
    name = " ".join(reversed(collected))
    return clean_object_name(name)

# 提取颜色标注的 point/bbox
def extract_color_mentions(question):
    matches = []
    for match in MENTION_RE.finditer(question):
        color = normalize_color(match.group(1)) # 颜色
        kind = match.group(2).lower() # 类型 point/bbox
        obj_name = extract_object_name_from_left_context(question[: match.start()]) #只是为了结果可读性/可视化标签（写进 name 字段、画在图上），不会作为文本提示传入 SAM2。SAM2 只吃坐标提示
        matches.append((obj_name, color, kind))
    return matches


def extract_object_name_no_color(question):
    question = question.replace("’", "'")
    match = re.search(r"(length|width|height|size)\b", question, flags=re.IGNORECASE)
    if match:
        name = extract_object_name_from_left_context(question[: match.start()])
        if name:
            return name
    patterns = [
        r"for\s+([^,?.\n]+)",
        r"of\s+([^,?.\n]+)",
        r"center\s+of\s+([^,?.\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, question, flags=re.IGNORECASE)
        if match:
            return clean_object_name(match.group(1))
    return "object"


def resolve_image_path(sample):
    image_list = sample.get("image", [])
    annotated = image_list[0] if image_list else None
    raw_candidate = None
    ply_list = sample.get("ply_path")
    if isinstance(ply_list, list) and ply_list and isinstance(ply_list[0], str):
        raw_candidate = ply_list[0]
        if raw_candidate.endswith("_full_fov_camera.ply"):
            raw_candidate = raw_candidate.replace("_full_fov_camera.ply", ".jpg")
    if raw_candidate and os.path.isfile(raw_candidate):
        return raw_candidate
    if annotated and os.path.isfile(annotated):
        return annotated
    return annotated or raw_candidate


def extract_scene_dir(sample_id):
    if isinstance(sample_id, str) and "_" in sample_id:
        return sample_id.rsplit("_", 1)[0]
    return "unknown"


def sanitize_filename(text):
    if text is None:
        return "unknown"
    text = str(text)
    text = text.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]", "_", text)


def get_first_coords(value, expected_len):
    if not isinstance(value, list) or not value:
        return None
    item = value[0]
    if not isinstance(item, list) or len(item) != expected_len:
        return None
    return tuple(float(v) for v in item)


def get_question(sample):
    conversations = sample.get("conversations", [])
    for conv in conversations:
        if conv.get("from") == "human":
            return conv.get("value", "")
    return ""

'''
输入：一个 sample
输出：一个列表，每个元素是一个 dict，包含：
    name: <对象名>
    color: 颜色
    prompt_type: "point" 或 "bbox"
    coords:<原始坐标(0..1000坐标系)>
    key: 标注字段名（例如 "red_point" 或 "green_bbox"）
'''
def extract_prompts_from_sample(sample):
    question = get_question(sample) # 问题文本
    prompts = []
    matches = extract_color_mentions(question) # 提取颜色标注的 point/bbox
    used_keys = set()

    for obj_name, color, kind in matches:
        key = f"{color}_{kind}"
        expected_len = 2 if kind == "point" else 4
        coords = get_first_coords(sample.get(key), expected_len) # 提取坐标(这里暂时是0-1000的归一化坐标系)
        if coords:
            prompts.append(
                {
                    "name": obj_name or color,
                    "color": color,
                    "prompt_type": kind,
                    "coords": coords,
                    "key": key,
                }
            )
            used_keys.add(key)

    if not prompts:
        if sample.get("type") == "spatial_volume_infer": # 如果样本类型是 spatial_volume_infer，则提取对象名和坐标
            obj_name = extract_object_name_no_color(question)
            coords = get_first_coords(sample.get("red_bbox"), 4) # 提取红色框的坐标
            if coords:
                prompts.append(
                    {
                        "name": obj_name,
                        "color": "red",
                        "prompt_type": "bbox",
                        "coords": coords,
                        "key": "red_bbox",
                    }
                )

    if not prompts:
        for key, val in sample.items():
            if key in used_keys:
                continue
            if key.endswith("_point"):
                coords = get_first_coords(val, 2)
                if coords:
                    color = normalize_color(key.replace("_point", ""))
                    prompts.append(
                        {
                            "name": color,
                            "color": color,
                            "prompt_type": "point",
                            "coords": coords,
                            "key": key,
                        }
                    )
            elif key.endswith("_bbox"):
                coords = get_first_coords(val, 4)
                if coords:
                    color = normalize_color(key.replace("_bbox", ""))
                    prompts.append(
                        {
                            "name": color,
                            "color": color,
                            "prompt_type": "bbox",
                            "coords": coords,
                            "key": key,
                        }
                    )

    return prompts


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image, np.array(image)


def save_mask_to_file(mask, path, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "png":
        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        img.save(path)
    elif fmt == "npy":
        np.save(path, mask.astype(np.uint8))
    else:
        raise ValueError(f"Unsupported mask format: {fmt}")


def clamp_point(point, width, height):
    x, y = point
    return (
        max(0.0, min(float(width - 1), float(x))),
        max(0.0, min(float(height - 1), float(y))),
    )


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), float(x1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    y2 = max(0.0, min(float(height - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def rle_encode(mask):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    height, width = mask.shape
    pixels = mask.T.flatten()  # Fortran order for COCO compatibility
    counts = []
    last_val = 0
    run_len = 0
    for val in pixels:
        if val == last_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            last_val = val
    counts.append(run_len)
    return {"size": [height, width], "counts": counts}


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


@contextlib.contextmanager
def inference_context(device):
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def map_yaml_name_to_config(name):
    if name.startswith("sam2.1_"):
        return f"configs/sam2.1/{name}"
    if name.startswith("sam2_"):
        return f"configs/sam2/{name}"
    return None


def infer_config_from_ckpt(ckpt_name):
    name = ckpt_name.lower()
    is_21 = "sam2.1" in name
    if "hiera_large" in name or "hiera_l" in name:
        yaml_name = "sam2.1_hiera_l.yaml" if is_21 else "sam2_hiera_l.yaml"
    elif "hiera_base_plus" in name or "hiera_b+" in name:
        yaml_name = "sam2.1_hiera_b+.yaml" if is_21 else "sam2_hiera_b+.yaml"
    elif "hiera_small" in name or "hiera_s" in name:
        yaml_name = "sam2.1_hiera_s.yaml" if is_21 else "sam2_hiera_s.yaml"
    elif "hiera_tiny" in name or "hiera_t" in name:
        yaml_name = "sam2.1_hiera_t.yaml" if is_21 else "sam2_hiera_t.yaml"
    else:
        return None
    return map_yaml_name_to_config(yaml_name)


def resolve_local_model_files(model_path):
    ckpt_path = None
    if os.path.isfile(model_path):
        ckpt_path = model_path
        search_dir = os.path.dirname(model_path)
    elif os.path.isdir(model_path):
        search_dir = model_path
    else:
        return None, None

    try:
        yaml_files = [name for name in os.listdir(search_dir) if name.endswith(".yaml")]
        pt_files = [name for name in os.listdir(search_dir) if name.endswith(".pt")]
    except FileNotFoundError:
        return None, None

    if ckpt_path is None:
        if not pt_files:
            return None, None
        preferred = [name for name in pt_files if "sam2.1_hiera_large" in name]
        if preferred:
            pt_files = preferred
        pt_files = sorted(pt_files)
        ckpt_path = os.path.join(search_dir, pt_files[0])

    config_name = None
    if yaml_files:
        for name in (
            "sam2.1_hiera_l.yaml",
            "sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_s.yaml",
            "sam2.1_hiera_t.yaml",
            "sam2_hiera_l.yaml",
            "sam2_hiera_b+.yaml",
            "sam2_hiera_s.yaml",
            "sam2_hiera_t.yaml",
        ):
            if name in yaml_files:
                config_name = map_yaml_name_to_config(name)
                break
        if config_name is None:
            yaml_files = sorted(yaml_files)
            config_name = map_yaml_name_to_config(yaml_files[0])

    if config_name is None and ckpt_path:
        config_name = infer_config_from_ckpt(os.path.basename(ckpt_path))

    return ckpt_path, config_name


class Sam2Segmenter:
    def __init__(self, model_path, device="cuda", config_name=None):
        if SAM2ImagePredictor is None:
            raise RuntimeError(
                f"SAM2ImagePredictor import failed: {_IMPORT_ERROR}"
            )
        self.device = resolve_device(device)
        if os.path.exists(model_path):
            if build_sam2 is None:
                raise RuntimeError(
                    f"SAM2 build_sam2 import failed: {_IMPORT_ERROR}"
                )
            ckpt_path, config_name = resolve_local_model_files(model_path)
            if not ckpt_path or not config_name:
                raise RuntimeError(
                    "本地模型加载失败：未找到 .pt 或 .yaml 配置文件。"
                )
            sam_model = build_sam2(
                config_file=config_name,
                ckpt_path=ckpt_path,
                device=self.device,
            )
            self.predictor = SAM2ImagePredictor(sam_model)
        else:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_path)
        self._move_model()

    def _move_model(self):
        if hasattr(self.predictor, "model"):
            try:
                self.predictor.model.to(self.device)
            except Exception:
                pass
        elif hasattr(self.predictor, "to"):
            try:
                self.predictor.to(self.device)
            except Exception:
                pass

    def segment(self, image_np, prompts, image_size):
        width, height = image_size
        self.predictor.set_image(image_np) # 为当前图像建立特征缓存
        results = []

        for prompt in prompts: # 遍历每个 prompt
            prompt_type = prompt["prompt_type"]
            coords = prompt["coords"]
            point_coords = None
            point_labels = None
            box = None
            multimask_output = True

            '''
                point_coords=[[sx, sy]]
                point_labels=[1]
                box=None
            '''
            if prompt_type == "point": # 如果 prompt 类型是 point，则转换为 SAM2 的坐标系
                sx, sy = scale_point(coords, width, height)
                sx, sy = clamp_point((sx, sy), width, height) # 变成像素坐标
                point_coords = np.array([[sx, sy]], dtype=np.float32) # 组装成 SAM2 需要的 numpy
                point_labels = np.array([1], dtype=np.int32)
                scaled_point = [float(sx), float(sy)]
                scaled_bbox = None
            else: # 如果 prompt 类型是 bbox，则转换为 SAM2 的坐标系
                sx1, sy1, sx2, sy2 = scale_bbox(coords, width, height)
                sx1, sy1, sx2, sy2 = clamp_bbox((sx1, sy1, sx2, sy2), width, height)
                scaled_bbox = [float(sx1), float(sy1), float(sx2), float(sy2)]
                box = np.array([scaled_bbox], dtype=np.float32)
                scaled_point = None
                multimask_output = False

            with inference_context(self.device): # 使用推理上下文
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output, # point 多候选，bbox 单候选
                )

            if torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.detach().cpu().numpy()

            if masks is None or len(masks) == 0:
                continue
            
            # 选择得分最高的 mask
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx].astype(bool)
            best_score = float(scores[best_idx])

            results.append(
                {
                    "name": prompt["name"], # 对象名
                    "color": prompt["color"], # 颜色
                    "prompt_type": prompt_type, # 提示类型
                    "prompt_key": prompt["key"], # 提示关键字  
                    "coords_raw": [float(v) for v in coords], # 原始坐标(0-1000)
                    "point_xy": scaled_point, # 点坐标
                    "bbox_xyxy": scaled_bbox, # 框坐标
                    "score": best_score, # 得分
                    "mask": best_mask, # 掩码
                }
            )

        return results


def resolve_device(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, falling back to cpu.", file=sys.stderr)
        return "cpu"
    return device


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def resolve_rank_out_jsonl(out_jsonl, rank, world_size):
    if world_size <= 1:
        return out_jsonl
    return f"{out_jsonl}.rank{rank}"


def merge_rank_outputs(out_jsonl, world_size):
    merged_path = out_jsonl
    tmp_path = out_jsonl + ".merged.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f_out:
        for rank in range(world_size):
            shard_path = f"{out_jsonl}.rank{rank}"
            if not os.path.isfile(shard_path):
                continue
            with open(shard_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
    os.replace(tmp_path, merged_path)

# 叠加 mask 到图像上
def overlay_masks(image, results, alpha=0.5):
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size

    for item in results:
        mask = item["mask"]
        color = COLOR_MAP.get(item["color"], (255, 255, 255))
        mask_img = Image.fromarray((mask * int(255 * alpha)).astype(np.uint8), mode="L")
        color_img = Image.new("RGBA", (width, height), color + (0,))
        color_img.putalpha(mask_img)
        overlay = Image.alpha_composite(overlay, color_img)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(composed)
    for item in results:
        color = COLOR_MAP.get(item["color"], (255, 255, 255))
        if item["prompt_type"] == "point" and item["point_xy"]:
            x, y = item["point_xy"]
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
            draw.text((x + 8, y + 8), item["name"], fill=color)
        elif item["prompt_type"] == "bbox" and item["bbox_xyxy"]:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 4, y1 + 4), item["name"], fill=color)
    return composed

# 要添加的sam2_seg字段内容
def serialize_results(results, image_size, sample_id, mask_root, mask_format):
    scene_dir = extract_scene_dir(sample_id)
    sample_id_safe = sanitize_filename(sample_id)
    mask_ext = ".png" if mask_format == "png" else ".npy"
    out = {
        "image_size": [int(image_size[0]), int(image_size[1])], # 图像宽高
        "mask_root": mask_root,
        "mask_format": mask_format,
        "objects": [], # 对应本 sample 中每一个 prompt（point/bbox）跑出来的“最佳 mask”结果
    }
    for idx, item in enumerate(results):
        mask = item["mask"]
        color = sanitize_filename(item["color"])
        ptype = sanitize_filename(item["prompt_type"])
        filename = f"{sample_id_safe}_{color}_{ptype}_{idx}{mask_ext}"
        rel_dir = sanitize_filename(scene_dir)
        rel_path = os.path.join(rel_dir, filename)
        abs_path = os.path.join(mask_root, rel_path)
        save_mask_to_file(mask, abs_path, mask_format)
        obj = {
            "name": item["name"], # 对象名
            "color": item["color"], # 颜色
            "prompt_type": item["prompt_type"], # 提示类型
            "prompt_key": item["prompt_key"], # 提示关键字
            "coords_raw": item["coords_raw"], # 原始 0..1000 坐标系里的输入坐标
            "point_xy": item["point_xy"], # 映射到像素坐标系后的点（float），仅 point 有
            "bbox_xyxy": item["bbox_xyxy"], # 映射到像素坐标系后的框（float），仅 bbox 有
            "score": item["score"], # 得分
            "mask_path": rel_path, # 掩码文件相对路径
            "mask_bbox": mask_to_bbox(mask), # 由 mask 反推的紧致外接框（int）
            "mask_area": int(mask.sum()), # mask 中 True 像素数（int）
        }
        out["objects"].append(obj)
    return out


def find_sample_by_id(jsonl_path, sample_id):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id") == sample_id:
                return obj
    return None


def process_single_sample(segmenter, sample, vis_out=None, mask_root="./sam2_masks", mask_format="png"):
    image_path = resolve_image_path(sample)
    if not image_path:
        raise RuntimeError("Sample has no image field.")
    sample_id = sample.get("id", "unknown")
    image, image_np = load_image(image_path)
    prompts = extract_prompts_from_sample(sample)
    results = segmenter.segment(image_np, prompts, image.size)
    if vis_out:
        vis_img = overlay_masks(image, results)
        vis_img.save(vis_out)
    seg = serialize_results(results, image.size, sample_id, mask_root, mask_format)
    return results, image.size, seg


def load_processed_ids(out_jsonl):
    processed = set()
    if not out_jsonl or not os.path.isfile(out_jsonl):
        return processed
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if sid is not None and "sam2_seg" in obj:
                processed.add(sid)
    return processed


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


def default_out_path(src_jsonl, suffix):
    base, ext = os.path.splitext(src_jsonl)
    if ext.lower() != ".jsonl":
        return src_jsonl + suffix + ".jsonl"
    return base + suffix + ext


def split_sam2_jsonl(
    src_jsonl,
    train_jsonl,
    val_jsonl,
    train_out,
    val_out,
    keep_unknown=False,
):
    train_ids = load_ids(train_jsonl)
    val_ids = load_ids(val_jsonl)
    total = 0
    train_cnt = 0
    val_cnt = 0
    unknown_cnt = 0
    progress = tqdm(desc="splitting", unit="lines") if tqdm else None
    with open(src_jsonl, "r", encoding="utf-8") as f_in, open(
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
    return total, train_cnt, val_cnt, unknown_cnt


def count_lines_for_rank(jsonl_path, rank=0, world_size=1):
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f):
            if world_size <= 1 or (i % world_size) == rank:
                total += 1
    return total


def process_jsonl(
    segmenter,
    jsonl_path,
    out_jsonl,
    mask_root="./sam2_masks",
    mask_format="png",
    max_samples=None,
    overwrite=False,
    rank=0,
    world_size=1,
):
    total = 0
    kept = 0
    processed_ids = set()
    if out_jsonl and os.path.isfile(out_jsonl) and not overwrite:
        processed_ids = load_processed_ids(out_jsonl)
    write_mode = "a" if (out_jsonl and os.path.isfile(out_jsonl) and not overwrite) else "w"
    total_lines = count_lines_for_rank(jsonl_path, rank=rank, world_size=world_size)
    progress = tqdm(total=total_lines, desc="processing", unit="lines") if tqdm else None
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(
        out_jsonl, write_mode, encoding="utf-8"
    ) as f_out:
        for line_idx, line in enumerate(f_in):
            if world_size > 1 and (line_idx % world_size) != rank:
                continue
            if not line.strip():
                f_out.write("\n")
                if progress is not None:
                    progress.update(1)
                continue
            sample = json.loads(line)
            total += 1
            sample_id = sample.get("id")
            if sample_id in processed_ids:
                if progress is not None:
                    progress.update(1)
                continue
            image_path = resolve_image_path(sample)
            if not image_path:
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if progress is not None:
                    progress.update(1)
                continue
            image, image_np = load_image(image_path)
            prompts = extract_prompts_from_sample(sample)
            results = segmenter.segment(image_np, prompts, image.size)
            sample["sam2_seg"] = serialize_results(
                results, image.size, sample_id, mask_root, mask_format
            )
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1
            for item in results:
                pname = item.get("name")
                pcolor = item.get("color")
                ptype = item.get("prompt_type")
                pscore = item.get("score")
                print(
                    f"[{sample_id}] {pname} ({pcolor} {ptype}) score={pscore:.4f}",
                    flush=True,
                )
            if progress is not None:
                progress.update(1)
            if max_samples is not None and kept >= max_samples:
                break
    if progress is not None:
        progress.close()
    return total, kept


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation for QA jsonl with point/bbox prompts."
    )
    parser.add_argument("--jsonl", default=None, help="Input jsonl path")
    parser.add_argument(
        "--model",
        default="/data3/lxp/Models/sam2.1-hiera-large",
        help="SAM2 model path",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--out-jsonl", help="Output jsonl with sam2_seg field")
    parser.add_argument(
        "--mask-dir",
        default="/data/dsq/ScanNet/qa_jsonl/images/sam2_masks",
        help="Directory to store mask files",
    )
    parser.add_argument(
        "--mask-format",
        default="png",
        choices=["png", "npy"],
        help="Mask file format (png or npy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output jsonl (disable resume)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging rank outputs when using torchrun",
    )
    parser.add_argument(
        "--split-src",
        default=None,
        help="Split an existing sam2 jsonl by id (e.g. all.sam2.jsonl)",
    )
    parser.add_argument(
        "--train-jsonl",
        default=None,
        help="Train jsonl with ids (e.g. train.jsonl)",
    )
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help="Val jsonl with ids (e.g. val.jsonl)",
    )
    parser.add_argument(
        "--train-out",
        default=None,
        help="Output train sam2 jsonl path",
    )
    parser.add_argument(
        "--val-out",
        default=None,
        help="Output val sam2 jsonl path",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep ids not in train/val (write to train output)",
    )
    parser.add_argument(
        "--vis-id", help="Sample id for visualization (optional)"
    )
    parser.add_argument(
        "--vis-out",
        default=None,
        help="Visualization output path (default: ./<id>_sam2.png)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N samples (debug)",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.split_src:
        if not args.train_jsonl or not args.val_jsonl:
            raise RuntimeError("Split mode requires --train-jsonl and --val-jsonl.")
        if not os.path.isfile(args.split_src):
            raise RuntimeError(f"split-src not found: {args.split_src}")
        if not os.path.isfile(args.train_jsonl):
            raise RuntimeError(f"train jsonl not found: {args.train_jsonl}")
        if not os.path.isfile(args.val_jsonl):
            raise RuntimeError(f"val jsonl not found: {args.val_jsonl}")
        train_out = args.train_out or default_out_path(args.train_jsonl, ".sam2")
        val_out = args.val_out or default_out_path(args.val_jsonl, ".sam2")
        total, train_cnt, val_cnt, unknown_cnt = split_sam2_jsonl(
            args.split_src,
            args.train_jsonl,
            args.val_jsonl,
            train_out,
            val_out,
            keep_unknown=args.keep_unknown,
        )
        print(
            f"Split done: total={total}, train={train_cnt}, val={val_cnt}, unknown={unknown_cnt}"
        )
        print(f"Train out: {train_out}")
        print(f"Val out: {val_out}")
        return

    if not args.jsonl or not os.path.isfile(args.jsonl):
        raise RuntimeError(f"jsonl not found: {args.jsonl}")

    rank, world_size, local_rank, distributed = init_distributed()
    device = args.device
    if distributed and torch.cuda.is_available():
        device = "cuda"
    segmenter = Sam2Segmenter(args.model, device=device)

    if args.vis_id:
        sample = find_sample_by_id(args.jsonl, args.vis_id)
        if sample is None:
            raise RuntimeError(f"Sample id not found: {args.vis_id}")
        vis_out = args.vis_out or f"{args.vis_id}_sam2.png"
        results, image_size, _ = process_single_sample(
            segmenter,
            sample,
            vis_out,
            mask_root=args.mask_dir,
            mask_format=args.mask_format,
        )
        print(f"Saved visualization: {vis_out}")
        print(f"Objects segmented: {len(results)}")
        for item in results:
            print(
                f"- {item['name']} ({item['color']} {item['prompt_type']}): score={item['score']:.4f}"
            )

    if args.out_jsonl:
        out_jsonl_rank = resolve_rank_out_jsonl(
            args.out_jsonl, rank=rank, world_size=world_size
        )
        total, kept = process_jsonl(
            segmenter,
            args.jsonl,
            out_jsonl_rank,
            mask_root=args.mask_dir,
            mask_format=args.mask_format,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
            rank=rank,
            world_size=world_size,
        )
        print(f"[rank {rank}] Processed samples: {total}, written: {kept}")
        if distributed:
            dist.barrier()
            if rank == 0 and not args.no_merge:
                merge_rank_outputs(args.out_jsonl, world_size)
                print(f"Merged rank outputs -> {args.out_jsonl}")
    elif not args.vis_id:
        print("Nothing to do: set --vis-id and/or --out-jsonl.")


if __name__ == "__main__":
    main()
