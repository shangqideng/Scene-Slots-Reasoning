"""
从冷启动 jsonl 中按 type 抽样，构造 GRPO 训练用的数据集。

输入（默认）:
  训练全集: /data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl
  验证集:   /data/dsq/ScanNet/qa_jsonl/val.jsonl   （用于去重过滤，防止数据泄露）

输出（默认）:
  训练集: /data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl
  验证集: /data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl

只保留以下类型的样本（按 json 字段 `type`）:
  - distance_infer_center_oo
  - obj_spatial_relation_oo
  - spatial_imagination_oc
  - spatial_imagination_oo
  - depth_prediction_oc
  - depth_prediction_oo
  - distance_prediction_oc
  - distance_prediction_oo

并组织为 swift GRPO 所需格式，每条样本结构类似:
{
  "images": ["/path/to/image.jpg"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "<query with <think>/<answer> instruction>"},
      ],
    }
  ],
  "solution": "<answer> 2.9 </answer>",
  "type": "distance_infer_center_oc",
  "id": "scene0001_00_43"
}

使用示例:
  python RL/build_grpo_dataset.py \
    --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl \
    --output-jsonl /data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl \
    --val-output-jsonl /data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl \
    --max-distance-oo 1000 \
    --max-obj-rel 1000 \
    --max-spatial-oc 1000 \
    --max-spatial-oo 1000 \
    --max-depth-oc 1000 \
    --max-depth-oo 1000 \
    --max-distance-pred-oc 1000 \
    --max-distance-pred-oo 1000 \
    --max-val-distance-oo 200 \
    --max-val-obj-rel 200 \
    --max-val-spatial-oc 200 \
    --max-val-spatial-oo 200 \
    --max-val-depth-oc 200 \
    --max-val-depth-oo 200 \
    --max-val-distance-pred-oc 200 \
    --max-val-distance-pred-oo 200

GRPO train dataset saved to: /data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl
GRPO val dataset saved to:   /data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl
  Total input lines: 27112
  Usable candidates (target types & valid, excl. test): 11148
  Excluded test ids: 1800
  Final selected for training: 3567
  Final selected for validation: 400

  python RL/build_grpo_dataset.py \
    --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.filtered_by_llm.jsonl \
    --output-jsonl /data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.modified_ds.filtered_by_llm.jsonl \
    --val-output-jsonl /data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.modified_ds.filtered_by_llm.jsonl \
    --max-distance-oo 2000 \
    --max-obj-rel 2000 \
    --max-spatial-oc 1000 \
    --max-spatial-oo 2000 \
    --max-depth-oc 2000 \
    --max-depth-oo 2000 \
    --max-distance-pred-oc 2000 \
    --max-distance-pred-oo 2000 \
    --max-val-distance-oo 0 \
    --max-val-obj-rel 0 \
    --max-val-spatial-oc 0 \
    --max-val-spatial-oo 0 \
    --max-val-depth-oc 0 \
    --max-val-depth-oo 0 \
    --max-val-distance-pred-oc 0 \
    --max-val-distance-pred-oo 0
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple


TARGET_TYPES = [
    "distance_infer_center_oo",
    "obj_spatial_relation_oo",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo"
]

# 默认每个任务的训练采样数（适配 Qwen2.5-VL-3B-Instruct 做 GRPO，避免数据量过大）
DEFAULT_MAX_PER_TYPE = {
    "distance_infer_center_oo": 500,
    "obj_spatial_relation_oo": 500,
    "spatial_imagination_oc": 500,
    "spatial_imagination_oo": 500,
    "depth_prediction_oc": 500,
    "depth_prediction_oo": 500,
    "distance_prediction_oc": 500,
    "distance_prediction_oo": 500,
}

# 默认每个任务的验证采样数（用于 GRPO 调参的小验证集）
DEFAULT_VAL_MAX_PER_TYPE = {
    "distance_infer_center_oo": 50,
    "obj_spatial_relation_oo": 50,
    "spatial_imagination_oc": 50,
    "spatial_imagination_oo": 50,
    "depth_prediction_oc": 50,
    "depth_prediction_oo": 50,
    "distance_prediction_oc": 50,
    "distance_prediction_oo": 50,
}


# rationale 过滤与优先级相关配置
# - 仅保留 rationale 单词数不超过该阈值的样本
MAX_RATIONALE_WORDS = 1000


def get_qa(sample: Dict[str, Any]) -> Tuple[str | None, str | None]:
    """从样本中提取 question 和 answer，兼容 conversations / question+answer 两种格式。"""
    # 优先从 conversations 中解析
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    # 回退到显式字段
    if question is None or answer is None:
        question = sample.get("question", question)
        answer = sample.get("answer", answer)
    if question is None or answer is None:
        return None, None
    return str(question).strip(), str(answer).strip()


def _get_rationale_word_len(sample: Dict[str, Any]) -> int:
    """统计 rationale 的单词数（按空白切分），缺失或非字符串时返回 0。"""
    rationale = sample.get("rationale")
    if not isinstance(rationale, str):
        return 0
    # 简单按空白切分统计“词”数量，适用于英文/空格分隔的场景
    return len(rationale.split())


def _ensure_image_list(image_field: Any) -> List[str]:
    """将 image 字段统一转为 list[str]."""
    if not image_field:
        return []
    if isinstance(image_field, str):
        return [image_field]
    if isinstance(image_field, list):
        # 只保留字符串路径
        return [x for x in image_field if isinstance(x, str)]
    return []


def _build_query(sample: Dict[str, Any]) -> str:
    """根据 question + scene_slot 构造带 <think>/<answer> 指令的 query 文本。"""
    question, _ = get_qa(sample)
    if not question:
        question = ""
    scene_slot = sample.get("scene_slot")
    if scene_slot is not None:
        slot_text = json.dumps(scene_slot, ensure_ascii=False)
        query = (
            f"{question}\n\n"
            "Scene slot (coords: x-right, y-down, z-forward):\n"
            f"{slot_text}\n\n"
            "Output the thinking process in <think> </think> and the final answer "
            "in <answer> </answer> tags. "
            "Write all detailed reasoning steps ONLY inside <think> </think>. "
            "Write ONLY the final result inside <answer> </answer>. "
        )
    else:
        query = (
            f"{question}\n\n"
            "Output the thinking process in <think> </think> and the final answer "
            "in <answer> </answer> tags. "
            "Write all detailed reasoning steps ONLY inside <think> </think>. "
            "Write ONLY the final result inside <answer> </answer>. "
        )
    return query


def _convert_sample_to_grpo(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    """将原始样本转换为 GRPO 所需格式."""
    q, a = get_qa(sample)
    if not q or not a:
        return None

    images = _ensure_image_list(sample.get("image"))
    if not images:
        # GRPO 多模态任务需要图像
        return None

    query = _build_query(sample)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]

    # 将真值包装到 <answer> 标签中，方便奖励函数解析
    solution = f"<answer> {str(a).strip()} </answer>"

    grpo_sample: Dict[str, Any] = {
        "images": images,
        "messages": messages,
        "solution": solution,
    }

    # 保留 id/type 等信息，方便调试或分析
    if "id" in sample:
        grpo_sample["id"] = sample["id"]
    if "type" in sample:
        grpo_sample["type"] = sample["type"]
    if "scene_slot" in sample:
        grpo_sample["scene_slot"] = sample["scene_slot"]

    # 透传内部使用的 rationale 长度信息，后续用于按长度优先抽样
    if "_rationale_word_len" in sample:
        grpo_sample["_rationale_word_len"] = sample["_rationale_word_len"]

    return grpo_sample


def _sample_by_type(
    samples_by_type: Dict[str, List[Dict[str, Any]]],
    max_per_type: Dict[str, int],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for t, items in samples_by_type.items():
        n_total = len(items)
        n_max = max_per_type.get(t, 0)
        if n_max is None or n_max <= 0 or n_max >= n_total:
            chosen = items
        else:
            chosen = rng.sample(items, n_max)
        selected.extend(chosen)
    # 打乱混合后的顺序
    rng.shuffle(selected)
    return selected


def build_grpo_dataset(
    input_path: str,
    output_path: str,
    seed: int = 42,
    test_path: str | None = "/data/dsq/ScanNet/qa_jsonl/val.jsonl",
    val_output_path: str | None = "/data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl",
    max_distance_oo: int = DEFAULT_MAX_PER_TYPE["distance_infer_center_oo"],
    max_obj_rel: int = DEFAULT_MAX_PER_TYPE["obj_spatial_relation_oo"],
    max_spatial_oc: int = DEFAULT_MAX_PER_TYPE["spatial_imagination_oc"],
    max_spatial_oo: int = DEFAULT_MAX_PER_TYPE["spatial_imagination_oo"],
    max_depth_oc: int = DEFAULT_MAX_PER_TYPE["depth_prediction_oc"],
    max_depth_oo: int = DEFAULT_MAX_PER_TYPE["depth_prediction_oo"],
    max_distance_pred_oc: int = DEFAULT_MAX_PER_TYPE["distance_prediction_oc"],
    max_distance_pred_oo: int = DEFAULT_MAX_PER_TYPE["distance_prediction_oo"],
) -> None:
    """
    构造 GRPO 训练/验证子集:
    - 根据 test_path（例如 val.jsonl）中的 id 过滤掉测试集样本，防止数据泄露
    - 在剩余数据中先采样一小部分作为验证集，再采样训练集，二者互斥
    """
    rng = random.Random(seed)

    train_max_per_type = {
        "distance_infer_center_oo": max_distance_oo,
        "obj_spatial_relation_oo": max_obj_rel,
        "spatial_imagination_oc": max_spatial_oc,
        "spatial_imagination_oo": max_spatial_oo,
        "depth_prediction_oc": max_depth_oc,
        "depth_prediction_oo": max_depth_oo,
        "distance_prediction_oc": max_distance_pred_oc,
        "distance_prediction_oo": max_distance_pred_oo,
    }

    # 读取测试集 id 集合，用于去重
    test_ids: set[str] = set()
    if test_path and os.path.isfile(test_path):
        with open(test_path, "r", encoding="utf-8") as f_test:
            for line_num, line in enumerate(f_test, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = obj.get("id")
                if vid is not None:
                    test_ids.add(str(vid))
        print(f"Loaded {len(test_ids)} unique ids from test set: {test_path}")
    else:
        if test_path:
            print(f"Warning: test file not found, skip leakage filter: {test_path}")

    samples_by_type: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TARGET_TYPES}
    total_lines = 0
    used_lines = 0
    excluded_test = 0

    if not os.path.isfile(input_path):
        raise RuntimeError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
                continue

            # 根据 id 过滤掉测试集样本，防止泄露
            sid = obj.get("id")
            if sid is not None and str(sid) in test_ids:
                excluded_test += 1
                continue

            t = obj.get("type")
            if t not in TARGET_TYPES:
                continue

            # 根据 rationale 长度进行过滤与标注
            rationale_len = _get_rationale_word_len(obj)
            if rationale_len > MAX_RATIONALE_WORDS:
                # rationale 过长，直接丢弃该样本
                continue
            # 记录到样本中，后续按长度优先抽样
            obj["_rationale_word_len"] = rationale_len

            grpo_sample = _convert_sample_to_grpo(obj)
            if grpo_sample is None:
                continue

            samples_by_type[t].append(grpo_sample)
            used_lines += 1

    print("Loaded samples by type (train+val candidates, excluding test):")
    for t in TARGET_TYPES:
        print(f"  {t}: {len(samples_by_type[t])} candidates")

    # 先在每个类型内划分一小部分作为验证集，再从剩余样本中采样训练集
    val_max_per_type = DEFAULT_VAL_MAX_PER_TYPE
    train_selected_all: List[Dict[str, Any]] = []
    val_selected_all: List[Dict[str, Any]] = []

    for t, items in samples_by_type.items():
        if not items:
            continue
        # 按 rationale 长度从短到长排序，优先选择 rationale 短的样本
        items_sorted = sorted(
            items,
            key=lambda x: x.get("_rationale_word_len", 10**9),
        )
        # 验证集样本
        v_max = val_max_per_type.get(t, 0)
        v_n = min(v_max, len(items)) if v_max > 0 else 0
        val_items = items_sorted[:v_n]
        remaining = items_sorted[v_n:]

        # 训练集样本
        t_max = train_max_per_type.get(t, 0)
        if t_max is None or t_max <= 0 or t_max >= len(remaining):
            train_items = remaining
        else:
            # 仍然优先选择 rationale 较短的样本
            train_items = remaining[:t_max]

        val_selected_all.extend(val_items)
        train_selected_all.extend(train_items)

    # 打乱总体顺序（保证不同 type 之间混合，同时不影响“优先选短 rationale”的原则）
    rng.shuffle(val_selected_all)
    rng.shuffle(train_selected_all)

    # 写出前移除内部使用的字段
    for obj in val_selected_all:
        obj.pop("_rationale_word_len", None)
    for obj in train_selected_all:
        obj.pop("_rationale_word_len", None)

    # 写训练集
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for obj in train_selected_all:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 写验证集
    if val_output_path:
        val_dir = os.path.dirname(val_output_path)
        if val_dir and not os.path.exists(val_dir):
            os.makedirs(val_dir, exist_ok=True)
        with open(val_output_path, "w", encoding="utf-8") as f_val_out:
            for obj in val_selected_all:
                f_val_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nGRPO train dataset saved to: {output_path}")
    if val_output_path:
        print(f"GRPO val dataset saved to:   {val_output_path}")
    print(f"  Total input lines: {total_lines}")
    print(f"  Usable candidates (target types & valid, excl. test): {used_lines}")
    print(f"  Excluded test ids: {excluded_test}")
    print(f"  Final selected for training: {len(train_selected_all)}")
    print(f"  Final selected for validation: {len(val_selected_all)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GRPO subset dataset from cold-start jsonl")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="/data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl",
        help="Input cold-start jsonl path (with question/answer/scene_slot/image/type).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl",
        help="Output GRPO train jsonl path.",
    )
    parser.add_argument(
        "--val-output-jsonl",
        type=str,
        default="/data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl",
        help="Output GRPO validation jsonl path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val-jsonl",
        type=str,
        default="/data/dsq/ScanNet/qa_jsonl/val.jsonl",
        help="Test jsonl used for id-based filtering to avoid leakage. "
             "Set to empty string to disable filtering.",
    )
    parser.add_argument(
        "--max-distance-oo",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["distance_infer_center_oo"],
        help="Max samples for type distance_infer_center_oo (0 = use all).",
    )
    parser.add_argument(
        "--max-obj-rel",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["obj_spatial_relation_oo"],
        help="Max samples for type obj_spatial_relation_oo (0 = use all).",
    )
    parser.add_argument(
        "--max-spatial-oc",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["spatial_imagination_oc"],
        help="Max samples for type spatial_imagination_oc (0 = use all).",
    )
    parser.add_argument(
        "--max-spatial-oo",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["spatial_imagination_oo"],
        help="Max samples for type spatial_imagination_oo (0 = use all).",
    )
    parser.add_argument(
        "--max-depth-oc",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["depth_prediction_oc"],
        help="Max samples for type depth_prediction_oc (0 = use all).",
    )
    parser.add_argument(
        "--max-depth-oo",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["depth_prediction_oo"],
        help="Max samples for type depth_prediction_oo (0 = use all).",
    )
    parser.add_argument(
        "--max-distance-pred-oc",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["distance_prediction_oc"],
        help="Max samples for type distance_prediction_oc (0 = use all).",
    )
    parser.add_argument(
        "--max-distance-pred-oo",
        type=int,
        default=DEFAULT_MAX_PER_TYPE["distance_prediction_oo"],
        help="Max samples for type distance_prediction_oo (0 = use all).",
    )
    parser.add_argument(
        "--max-val-distance-oo",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["distance_infer_center_oo"],
        help="Max validation samples for type distance_infer_center_oo.",
    )
    parser.add_argument(
        "--max-val-obj-rel",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["obj_spatial_relation_oo"],
        help="Max validation samples for type obj_spatial_relation_oo.",
    )
    parser.add_argument(
        "--max-val-spatial-oc",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["spatial_imagination_oc"],
        help="Max validation samples for type spatial_imagination_oc.",
    )
    parser.add_argument(
        "--max-val-spatial-oo",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["spatial_imagination_oo"],
        help="Max validation samples for type spatial_imagination_oo.",
    )
    parser.add_argument(
        "--max-val-depth-oc",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["depth_prediction_oc"],
        help="Max validation samples for type depth_prediction_oc.",
    )
    parser.add_argument(
        "--max-val-depth-oo",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["depth_prediction_oo"],
        help="Max validation samples for type depth_prediction_oo.",
    )
    parser.add_argument(
        "--max-val-distance-pred-oc",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["distance_prediction_oc"],
        help="Max validation samples for type distance_prediction_oc.",
    )
    parser.add_argument(
        "--max-val-distance-pred-oo",
        type=int,
        default=DEFAULT_VAL_MAX_PER_TYPE["distance_prediction_oo"],
        help="Max validation samples for type distance_prediction_oo.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_grpo_dataset(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        seed=args.seed,
        test_path=args.val_jsonl if args.val_jsonl else None,
        val_output_path=args.val_output_jsonl,
        max_distance_oo=args.max_distance_oo,
        max_obj_rel=args.max_obj_rel,
        max_spatial_oc=args.max_spatial_oc,
        max_spatial_oo=args.max_spatial_oo,
        max_depth_oc=args.max_depth_oc,
        max_depth_oo=args.max_depth_oo,
        max_distance_pred_oc=args.max_distance_pred_oc,
        max_distance_pred_oo=args.max_distance_pred_oo,
    )


if __name__ == "__main__":
    main()

