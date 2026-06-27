import json
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def select_short_rationales(
    input_path: str,
    output_path: str,
    k: int = 10,
    rationale_field: str = "rationale",
    type_field: str = "type",
) -> None:
    """
    从 input_path 读取 jsonl 文件，
    对每个 type 选出 rationale 最短的 k 条，并写入 output_path（jsonl）。
    """
    # 对每个类型维护一个大小不超过 k 的堆
    # 堆元素为 (neg_length, order, obj)
    # 使用负长度，这样堆顶是当前集合里 rationale 最长的样本，方便替换
    heaps: Dict[str, List[Tuple[int, int, Any]]] = defaultdict(list)

    order = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过非法行
                continue

            typ = obj.get(type_field)
            if typ is None:
                # 无 type 字段则跳过
                continue

            rationale = obj.get(rationale_field, "")
            length = len(rationale)

            # 使用负长度构成最小堆，堆顶为当前集合中 rationale 最长的样本
            key = -length
            heap = heaps[typ]

            order += 1
            entry = (key, order, obj)

            if len(heap) < k:
                heapq.heappush(heap, entry)
            else:
                # 只在新样本更短时替换当前最长样本
                # (key 越大，rationale 越短，因为 key = -length)
                if entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    # 收集所有类型的样本，并按 rationale 长度从短到长排序
    selected: List[Tuple[int, int, Any]] = []
    for typ, heap in heaps.items():
        # heap 中的 key 是负长度，这里转回正的长度
        for key, ord_, obj in heap:
            length = -key
            selected.append((length, ord_, obj))

    selected.sort(key=lambda x: (x[0], x[1]))  # 先按长度升序，再按出现顺序

    with open(output_path, "w", encoding="utf-8") as out_f:
        for _, _, obj in selected:
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_file = "/data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl"
    output_file = "/home/lxp/Ground_reasoning/all.question.rationale.short_10_per_type.jsonl"
    select_short_rationales(input_file, output_file, k=10)

