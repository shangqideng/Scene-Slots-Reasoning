"""
Swift GRPO 奖励函数插件：
- external_scanet_acc: 基于答案准确性的奖励函数

使用:
  --external_plugins RL/grpo_plugin_scanet.py \
  --reward_funcs external_scanet_acc format
"""

import os
import re
from typing import List

from swift.plugin import ORM, orms  # type: ignore


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# ----------------------
# 可调的评分阈值（便于让 RFT 的奖励更平滑、梯度更密集）
# ----------------------
# 单值数值题允许的最大相对误差；原来是 0.4，略偏严格，这里稍微放宽一点
SINGLE_NUM_MAX_REL_ERR = 0.6

# 三元数值题（L,W,H）允许的最大相对误差；原来是 0.30，这里也放宽一点
TRIPLE_NUM_MAX_REL_ERR = 0.4

# 调试输出相关：最多打印前 N 个样本的详细信息（只在 rank 0 打印）
DEBUG_PRINT_LIMIT = 10
_debug_print_count = 0


def _extract_answer_from_tagged(text: str) -> str:
    """从带 <answer> 标签的文本中抽取答案，不存在则返回原文本 strip 后."""
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _extract_numbers(s: str) -> list[float]:
    """从字符串中提取所有数字，返回 float 列表。"""
    if not s:
        return []
    nums: list[float] = []
    for m in _NUM_RE.findall(s):
        v = _parse_float(m)
        if v is not None:
            nums.append(v)
    return nums


def _extract_first_number(s: str) -> float | None:
    nums = _extract_numbers(s)
    return nums[0] if nums else None

def _extract_display_answer(s: str) -> str:
    """
    从完整的 completion 文本中，尽量提取一个简短的可读答案，用于调试打印：
    - 优先提取 <answer>...</answer> 内的内容；
    - 否则尝试从末尾提取 Yes/No、A/B/C/D 或最后一个数字；
    - 如果都失败，则返回原始文本（但这一般只在非常非结构化输出时出现）。
    """
    if not s:
        return ""

    # 1) 有 <answer> 标签时，直接取标签内容
    m = re.search(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    text = s.strip()

    # 2) 从后往前找 Yes / No
    yn_matches = list(re.finditer(r"\b(yes|no)\b", text, flags=re.IGNORECASE))
    if yn_matches:
        last = yn_matches[-1].group(1)
        return last.capitalize()

    # 3) 从后往前找 A/B/C/D 选项
    mc_matches = list(re.finditer(r"\b([A-D])\b", text, flags=re.IGNORECASE))
    if mc_matches:
        last = mc_matches[-1].group(1)
        return last.upper()

    # 4) 从后往前找最后一个数字
    num_matches = list(_NUM_RE.finditer(text))
    if num_matches:
        return num_matches[-1].group(0)

    # 5) 退化情况：取末尾一行（再 strip）作为显示答案
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text

def _extract_pred_for_reward(gt_str: str, completion: str) -> str:
    """
    根据真值类型（数字 / 选项 / yes-no / 文本），从完整 completion 中提取
    用于打分和调试的“预测答案”字符串。
    策略：
    1) 优先使用 <answer>...</answer> 或 "Answer: ..." 段落作为候选片段；
    2) 再根据 gt 类型分别提取：
       - 数值题：从候选片段中取数字（单值或三元）；
       - 选择题：从候选片段中取 A/B/C/D；
       - 判断题：从候选片段中取 Yes/No；
       - 其他：退化到通用的 _extract_display_answer。
    """
    text = (completion or "").strip()
    gt_str = (gt_str or "").strip()
    gt_lower = gt_str.lower()

    # 1) 先取候选片段：<answer>...</answer> 或 "Answer: ..." / "Final answer: ..."
    answer_segment = ""

    # 1.1 <answer> 标签
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        answer_segment = m.group(1).strip()
    else:
        # 1.2 Answer: / Final answer: 形式（取到行尾）
        m = re.search(
            r"(?i)(?:final\s+answer|answer)\s*[:：]\s*(.+)",
            text,
            flags=re.DOTALL,
        )
        if m:
            # 只取到首个换行，避免长段落
            seg = m.group(1).strip()
            answer_segment = seg.splitlines()[0].strip()

    # 如果没有专门的 answer 段，就退回全量文本
    if not answer_segment:
        answer_segment = text

    # 2) 根据 gt 类型分类处理
    gt_nums = _extract_numbers(gt_str)

    # 2.1 三元数值题 / 单值数值题
    if len(gt_nums) >= 1:
        # 在候选 answer 段中找数字；至少要有一个
        nums_in_seg = _NUM_RE.findall(answer_segment)
        if nums_in_seg:
            # 保留整个片段，后续用 _extract_numbers/_extract_first_number 再解析
            return answer_segment.strip()
        # 候选段没数字，则退回通用规则
        return _extract_display_answer(text)

    # 2.2 选择题 A/B/C/D
    if len(gt_str) == 1 and gt_str.upper() in {"A", "B", "C", "D"}:
        # 优先在 answer 段中找 A-D
        m = re.search(r"\b([A-D])\b", answer_segment, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # 再在全文中从后往前找 A-D
        mc_matches = list(re.finditer(r"\b([A-D])\b", text, flags=re.IGNORECASE))
        if mc_matches:
            return mc_matches[-1].group(1).upper()
        # 实在找不到，退化为通用规则
        return _extract_display_answer(text)

    # 2.3 Yes/No 判断题
    if gt_lower in {"yes", "no"}:
        # 优先在 answer 段中找 yes/no
        m = re.search(r"\b(yes|no)\b", answer_segment, flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        # 再在全文中从后往前找 yes/no
        yn_matches = list(re.finditer(r"\b(yes|no)\b", text, flags=re.IGNORECASE))
        if yn_matches:
            return yn_matches[-1].group(1).capitalize()
        # 退化为通用规则
        return _extract_display_answer(text)

    # 2.4 其他字符串题
    return _extract_display_answer(text)


def _score_single_numeric(gt: float, pred: float) -> float:
    """
    单个数值评分规则（与 9score_only_vllm.py 思路一致，阈值略放宽）:
    - rel_err = |pred-gt| / max(|gt|, 1e-6)
    - rel_err > SINGLE_NUM_MAX_REL_ERR -> 0
    - else: 1 - rel_err / SINGLE_NUM_MAX_REL_ERR
    """
    rel_err = abs(pred - gt) / max(abs(gt), 1e-6)
    if rel_err > SINGLE_NUM_MAX_REL_ERR:
        return 0.0
    return max(0.0, 1.0 - rel_err / SINGLE_NUM_MAX_REL_ERR)


def _score_triple_numeric(gt_nums: list[float], pred_nums: list[float]) -> float:
    """
    三个数值评分规则（参考 9score_only_vllm.py:80-94，阈值略放宽）:
    - 针对 spatial_volume_infer，格式: L,W,H（单位可认为统一）
    - 若 pred 无法解析出恰好 3 个数: 0
    - 对每一维 i:
        re_i = |p_i-g_i|/max(|g_i|,1e-6)
        如果 re_i > 0.30: s_i = 0
        否则: s_i = 1 - re_i/0.30
      score = (s_L + s_W + s_H)/3
    - 若预测满足 L <= W（违反 length > width 先验约束）: score *= 0.7
    """
    if len(gt_nums) != 3 or len(pred_nums) != 3:
        return 0.0
    s_list: list[float] = []
    for p, g in zip(pred_nums, gt_nums):
        re_i = abs(p - g) / max(abs(g), 1e-6)
        if re_i > TRIPLE_NUM_MAX_REL_ERR:
            s_i = 0.0
        else:
            s_i = max(0.0, 1.0 - re_i / TRIPLE_NUM_MAX_REL_ERR)
        s_list.append(s_i)
    score = sum(s_list) / 3.0
    # 约束: 预测的 L 应该 > W，否则惩罚
    pL, pW, _ = pred_nums
    if pL <= pW:
        score *= 0.7
    # 保证在 [0,1]
    return float(max(0.0, min(1.0, score)))


class ScanetAccuracyORM(ORM):
    """
    针对 ScanNet 3D 任务的简单准确性奖励函数。

    约定:
    - solution: 形如 "<answer> 2.9 </answer>" 或 "2.9" / "A" / "yes"
    - completion: 模型生成的文本，可以包含 <answer> 标签或自然语言

    评分策略:
    1) 若真值中只含 1 个数字（单值数值题，例如距离/深度）:
       - 从 completion 中抽取第一个数字
       - 使用相对误差: rel_err = |pred-gt| / max(|gt|, 1e-6)
       - 若 rel_err > 0.4: reward = 0.0
       - 否则: reward = 1 - rel_err / 0.4  (线性衰减到 0)
    2) 若真值中恰好包含 3 个数字（例如 spatial_volume_infer，L,W,H）:
       - 若 pred 不能解析出恰好 3 个数: reward = 0.0
       - 对每一维按 0.30 的相对误差阈值计算 s_i，最终 score = (s_L+s_W+s_H)/3
       - 若预测满足 L <= W，则 score *= 0.7
    3) 若真值为单个选项字母 A/B/C/D（选择题）:
       - 从 completion 中抽取第一个 A/B/C/D
       - 全等则 reward = 1.0，否则 0.0
    4) 若真值为 yes/no（判断题）:
       - 大小写不敏感匹配 yes/no，且不混用对立词
    5) 其他情况（如物体名称等）:
       - 将两者统一为小写去空格后比较:
         * 完全相等，或一方为另一方子串，则 reward = 1.0
         * 否则 reward = 0.0
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:  # type: ignore[override]
        rewards: List[float] = []
        global _debug_print_count
        for comp, sol in zip(completions, solution):
            comp = comp or ""
            sol = sol or ""

            # 解析真值（solution 始终是 <answer>...</answer>）
            gt = _extract_answer_from_tagged(sol)
            gt_str = gt.strip()

            # 预测答案：结合真值类型，从完整 completion 中抽取用于打分的答案字符串
            pred_str = _extract_pred_for_reward(gt_str, comp)

            # 仅在 rank 0 上打印前 DEBUG_PRINT_LIMIT 个样本的调试信息
            # if _debug_print_count < DEBUG_PRINT_LIMIT and os.environ.get("RANK", "0") == "0":
            #     idx = _debug_print_count + 1
            #     print("=" * 80)
            #     print(f"[ScanetAccuracyORM Debug] Sample {idx}/{DEBUG_PRINT_LIMIT}")
            #     print("[GT       ]", gt_str)
            #     print("[PRED_RAW ]", comp)
            #     print("[PRED_ANS ]", pred_str)
            #     print("=" * 80)
            #     _debug_print_count += 1

            reward = 0.0

            # 先基于真值解析数值信息
            gt_nums = _extract_numbers(gt_str)

            # 2) 三个数值（例如 spatial_volume_infer）
            if len(gt_nums) == 3:
                pred_nums = _extract_numbers(pred_str)
                reward = _score_triple_numeric(gt_nums, pred_nums)
                rewards.append(float(reward))
                continue

            # 1) 单个数值（常规距离/深度/标量问题）
            if len(gt_nums) == 1:
                gt_num = gt_nums[0]
                pred_num = _extract_first_number(pred_str)
                if pred_num is not None:
                    reward = _score_single_numeric(gt_num, pred_num)
                else:
                    reward = 0.0
                rewards.append(float(reward))
                continue

            # 选项题 A/B/C/D
            if len(gt_str) == 1 and gt_str.upper() in {"A", "B", "C", "D"}:
                m = re.search(r"\b([A-D])\b", pred_str, flags=re.IGNORECASE)
                if m and m.group(1).upper() == gt_str.upper():
                    reward = 1.0
                rewards.append(float(reward))
                continue

            # Yes/No 问题
            gt_lower = gt_str.lower()
            if gt_lower in {"yes", "no"}:
                pred_lower = pred_str.lower()
                if gt_lower == "yes" and "yes" in pred_lower and "no" not in pred_lower:
                    reward = 1.0
                elif gt_lower == "no" and "no" in pred_lower and "yes" not in pred_lower:
                    reward = 1.0
                rewards.append(float(reward))
                continue

            # 其他: 字符串/类别名称等，做宽松匹配
            norm_gt = gt_str.strip().lower()
            norm_pred = pred_str.strip().lower()
            if norm_gt and norm_pred:
                if norm_gt == norm_pred or norm_gt in norm_pred or norm_pred in norm_gt:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = 0.0
            rewards.append(float(reward))

        return rewards


# 注册到 swift 的 ORM 字典中，名称在 --reward_funcs 中使用
orms["external_scanet_acc"] = ScanetAccuracyORM


class FormatORM(ORM):
    """
    格式奖励函数：检查模型生成是否同时包含 <thinking> 和 <answer> 标签格式。
    
    奖励规则:
    - 如果completion中同时包含 <thinking>...</thinking> 和 <answer>...</answer> 标签: reward = 1.0
    - 如果只包含其中一个标签: reward = 0.5 (部分奖励)
    - 如果都不包含: reward = 0.0
    
    注意: 使用 DOTALL 标志以匹配跨行的标签内容。
    """
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:  # type: ignore[override]
        rewards: List[float] = []
        for comp in completions:
            comp = comp or ""

            # 检查是否包含格式正确的 <think> 或 <thinking> 标签
            has_thinking = bool(
                re.search(
                    r"<think>.*?</think>|<thinking>.*?</thinking>",
                    comp,
                    flags=re.DOTALL | re.IGNORECASE,
                )
            )

            # 检查是否包含格式正确的 <answer> 标签
            has_answer = bool(
                re.search(r"<answer>.*?</answer>", comp, flags=re.DOTALL | re.IGNORECASE)
            )
            
            # 根据是否同时包含两个标签给予奖励
            if has_thinking and has_answer:
                rewards.append(1.0)
            elif has_thinking or has_answer:
                rewards.append(0.5)  # 部分奖励
            else:
                rewards.append(0.0)
        
        return rewards


# 注册 format 奖励函数
# 如果swift库中已有format实现，这会覆盖它
orms["format"] = FormatORM

