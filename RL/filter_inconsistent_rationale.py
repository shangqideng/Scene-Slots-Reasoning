"""
使用本地 Qwen2.5-14B-Instruct-1M 模型，检查每个样本中
  - question（human value）
  - gpt answer（gpt value）
  - rationale
三者的一致性。

规则（交给 LLM 严格执行）：
- 判断题 / 选择题：如果 rationale 的推理结论与最终 gpt answer 明显不一致，则删除该样本；
- 数值题：如果 rationale 中给出的“正确数值”与 gpt answer 的偏差超过 70%（相对误差 > 0.7），则删除该样本；
- 解析失败或模型无法判断的样本，保守起见也删除。

输出：仅保留“通过一致性检查”的样本到新的 jsonl。
"""

import argparse
import json
import os
from ast import literal_eval
from typing import List, Tuple
import re

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCORE_LLM_PATH = "/data3/lxp/Models/Qwen2.5-14B-Instruct-1M"
DTYPE = torch.bfloat16


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] JSON decode error in {path}:{lineno}: {e}. "
                    f"Line content (truncated): {line[:200]!r}"
                )
                continue
    return data


def build_consistency_prompt(
    question: str, answer: str, rationale: str, tokenizer: AutoTokenizer
) -> str:
    """构造让 LLM 判断 rationale 与 gpt answer 是否一致的 prompt（不做长度控制）。"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict data cleaner for a 3D visual QA dataset. "
                "You must decide whether to KEEP or DELETE a sample based on "
                "the consistency between the question, the final answer, and the rationale.\n\n"
                "Rules:\n"
                "1) The sample contains: a question (with task description), a final short answer, "
                "   and a detailed rationale showing intermediate calculations or logic.\n"
                "2) For yes/no or multiple-choice questions (A/B/C/D etc.):\n"
                "   - If the rationale's reasoning clearly supports the final answer, mark consistent.\n"
                "   - If the rationale's reasoning clearly leads to a different conclusion or option, "
                "     mark inconsistent.\n"
                "3) For numeric questions (single numeric answer):\n"
                "   - The rationale usually contains a 'true' numeric result from calculation.\n"
                "   - Compute relative error = |answer - rationale_value| / max(|rationale_value|, 1e-6).\n"
                "   - If relative error > 0.7 (70%): mark inconsistent.\n"
                "   - Else: mark consistent.\n"
                "4) If you cannot confidently parse the rationale or judge consistency, "
                "   treat the sample as inconsistent (DELETE).\n\n"
                "Output format:\n"
                "Return ONLY a Python dict string like:\n"
                "  {'keep': True/False, 'reason': 'short explanation'}\n"
                "Do NOT output anything else."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is one sample:\n\n"
                f"Question:\n{question}\n\n"
                f"Final answer from model (gpt):\n{answer}\n\n"
                f"Rationale (step-by-step reasoning):\n{rationale}\n\n"
                "Decide if this sample should be kept or deleted based on the above rules.\n"
                "Remember: If you are not sure, set keep=False."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_consistency_prompt_with_truncation(
    question: str,
    answer: str,
    rationale: str,
    tokenizer: AutoTokenizer,
    max_model_len: int,
    reserve_tokens: int = 128,
) -> str:
    """
    构造 prompt，并在必要时截断 rationale，避免超过 max_model_len。
    这里用“先构造、再测长度、再截断”的迭代方式，确保安全。
    """
    # 最多截断多次，防止死循环
    for _ in range(10):
        prompt = build_consistency_prompt(question, answer, rationale, tokenizer)
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(input_ids) <= max_model_len - reserve_tokens:
            return prompt
        # 超长则截断 rationale：每次保留前 80% 字符
        if not rationale:
            return prompt
        new_len = max(1, int(len(rationale) * 0.8))
        rationale = rationale[:new_len]
    # 截断多次仍然过长，则直接返回最后一次结果（极少数情况）
    return build_consistency_prompt(question, answer, rationale, tokenizer)


_DICT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def parse_keep_output(text: str) -> Tuple[bool, bool]:
    """解析 LLM 输出，返回 (keep, ok)。如果解析失败，则 ok=False。"""
    if not text:
        return False, False
    m = _DICT_RE.search(text.strip())
    if not m:
        return False, False
    blob = m.group(0)
    try:
        obj = literal_eval(blob)
    except Exception:
        return False, False
    if not isinstance(obj, dict):
        return False, False
    if "keep" not in obj:
        return False, False
    keep = obj["keep"]
    if isinstance(keep, str):
        keep_str = keep.strip().lower()
        if keep_str in ("true", "yes", "keep"):
            return True, True
        if keep_str in ("false", "no", "delete", "drop"):
            return False, True
        return False, False
    if isinstance(keep, bool):
        return bool(keep), True
    return False, False


def filter_with_vllm(
    data: List[dict],
    tp: int,
    gpu_mem_util: float,
    max_model_len: int,
    batch_size: int,
    max_tokens: int,
) -> List[dict]:
    """使用本地 Qwen 模型对样本进行一致性过滤。"""
    tokenizer = AutoTokenizer.from_pretrained(SCORE_LLM_PATH)
    tokenizer.padding_side = "left"

    llm = LLM(
        model=SCORE_LLM_PATH,
        dtype=DTYPE,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        stop=None,
    )

    kept: List[dict] = []
    total = len(data)
    for start in range(0, total, batch_size):
        batch = data[start : start + batch_size]
        prompts: List[str] = []
        for sample in batch:
            conversations = sample.get("conversations", [])
            q = ""
            a = ""
            for item in conversations:
                if item.get("from") == "human" and not q:
                    q = str(item.get("value", "")).strip()
                elif item.get("from") == "gpt" and not a:
                    a = str(item.get("value", "")).strip()
            rationale = str(sample.get("rationale", "")).strip()
            if not q or not a or not rationale:
                # 基本字段缺失，直接视为不一致（不保留），但仍然送给 LLM 以保持流程统一
                prompt = build_consistency_prompt_with_truncation(
                    q or "(missing question)",
                    a or "(missing answer)",
                    rationale or "(missing rationale)",
                    tokenizer,
                    max_model_len,
                )
            else:
                prompt = build_consistency_prompt_with_truncation(
                    q,
                    a,
                    rationale,
                    tokenizer,
                    max_model_len,
                )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        for sample, out in zip(batch, outputs):
            gen_text = out.outputs[0].text if out.outputs else ""
            keep, ok = parse_keep_output(gen_text)
            # 解析失败或 keep=False -> 删除
            if ok and keep:
                kept.append(sample)

        print(f"[Filter] processed {min(start + batch_size, total)}/{total}, kept={len(kept)}")

    # 清理显存
    del llm, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return kept


def main():
    parser = argparse.ArgumentParser(
        description="Filter inconsistent rationale/answer samples using local Qwen2.5-14B with vLLM."
    )
    parser.add_argument(
        "--input-jsonl", type=str, required=True, help="Input jsonl file path."
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output jsonl file path for filtered samples.",
    )
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size.")
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.6,
        help="GPU memory utilization for vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max sequence length for the model.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for vLLM inference."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max generation length for vLLM.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_jsonl):
        raise RuntimeError(f"Input file not found: {args.input_jsonl}")

    data = load_jsonl(args.input_jsonl)
    print(f"[Filter] Loaded {len(data)} samples from {args.input_jsonl}")

    kept = filter_with_vllm(
        data=data,
        tp=args.tp,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    print(f"[Filter] Kept {len(kept)} / {len(data)} samples "
          f"({len(kept) / max(len(data), 1) * 100:.2f}%)")

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[Filter] Saved filtered samples to {args.output_jsonl}")


if __name__ == "__main__":
    main()

