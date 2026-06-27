'''
torchrun --nproc_per_node=8 RL/infer_rationale_sft.py \
  --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
  --lora-path /home/lxp/Ground_reasoning/Models/rationale_sft/v6-20260215-151015/checkpoint-3390 \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.sam2.jsonl \
  --pred-jsonl /data/dsq/ScanNet/qa_jsonl/infer/rationale_sft_v6/val.pred.jsonl \
  --use-scene-slot

用于推理训练好的rationale SFT模型，只提取answer部分。
参考8infer.py的结构，确保输出格式兼容9score_only_vllm.py。
'''
import os
import json
import gc
import argparse
import re

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


DTYPE = torch.bfloat16


def _sanitize_distributed_env():
    for k in [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
    ]:
        os.environ.pop(k, None)


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            try:
                dist.init_process_group(
                    backend=backend, init_method="env://", device_id=local_rank
                )
            except TypeError:
                dist.init_process_group(backend=backend, init_method="env://")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def safe_barrier(local_rank: int):
    if not dist.is_initialized():
        return
    try:
        dist.barrier(device_ids=[local_rank])
    except TypeError:
        dist.barrier()


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def build_qa(sample):
    """
    从样本中提取 question 和 answer，尽量与训练阶段的 get_qa 逻辑保持一致：
    1) 优先从 conversations 中解析；
    2) 若缺失则回退到顶层字段 question / answer。
    """
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    # 回退到显式字段（兼容可能存在的顶层 answer）
    if question is None:
        question = sample.get("question")
    if answer is None:
        answer = sample.get("answer")
    return question, answer


def get_scene_slot(sample):
    return sample.get("scene_slot")


def build_prompt(processor, question: str, scene_slot=None, use_scene_slot: bool = False) -> str:
    """
    构建推理提示，采用与 SFT 训练阶段一致的 <think>/<answer> 指令格式。
    虽然最终只保留 <answer> 中的答案用于评分，但这里仍允许模型输出完整思考过程。
    """
    if use_scene_slot:
        slot_text = json.dumps(scene_slot, ensure_ascii=False)
        text = (
            f"{question}\n\n"
            "Scene slot (coords: x-right, y-down, z-forward):\n"
            f"{slot_text}\n\n"
            "Output the thinking process in <think> </think> and the final answer "
            "in <answer> </answer> tags. "
            "Write all detailed reasoning steps ONLY inside <think> </think>. "
            "Write ONLY the final result inside <answer> </answer>."
        )
    else:
        text = (
            f"{question}\n\n"
            "Output the thinking process in <think> </think> and the final answer "
            "in <answer> </answer> tags. "
            "Write all detailed reasoning steps ONLY inside <think> </think>. "
            "Write ONLY the final result inside <answer> </answer>."
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_answer_from_response(response: str, question: str) -> str:
    """
    从模型生成的response中提取answer部分。
    由于训练时使用了不同的label mask，模型可能输出：
    1. 只有rationale（训练时rationale_only）
    2. 只有answer（训练时answer_only）
    3. rationale + answer（训练时both）
    
    我们需要优先提取answer部分，如果只有rationale则尝试从rationale中推断答案。
    """
    if not response:
        return ""
    
    # 清理response
    text = response.strip()

    # 策略0: 优先从 <answer>...</answer> 标签中提取（与训练时格式对齐）
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer
    
    # 策略1: 优先查找 "Answer: xxx" / "Final answer: xxx" 格式（兼容旧 prompt）
    # 匹配 "Answer:" 后的内容，直到文本结束或遇到明显的分隔符
    answer_patterns = [
        r"(?i)(?:final\s+answer|answer)\s*:\s*(.+?)(?:\n\n|\n\s*\n|$)",
        r"(?i)(?:final\s+answer|answer)\s*:\s*(.+)$",
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # 移除可能的后续内容（如额外的换行、空行等）
            answer = re.sub(r"\n\n+.*$", "", answer, flags=re.DOTALL).strip()
            if answer:
                return answer
    
    # 策略2: 如果response很短（<100字符），可能是直接答案
    if len(text) < 100:
        # 移除可能的question重复
        q = (question or "").strip()
        if q and text.startswith(q):
            text = text[len(q):].strip()
        # 移除可能的"assistant"等标记
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.lower() in ("assistant", "assistant:") or line.lower().startswith("assistant:"):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines).strip()
        if text:
            return text
    
    # 策略3: 如果response很长，可能是rationale，尝试从最后提取可能的答案
    # 查找最后一个包含数字、选项字母（A/B/C/D）、或短句子的段落
    lines = text.splitlines()
    
    # 从后往前查找包含 "Answer:" 或 "answer:" 的行
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if re.search(r"(?i)\banswer\s*:", line):
            # 找到包含 "Answer:" 的行，提取该行及之后的内容
            answer_lines = [line]
            for j in range(i + 1, len(lines)):
                answer_lines.append(lines[j])
            answer = "\n".join(answer_lines).strip()
            # 提取冒号后的内容
            match = re.search(r":\s*(.+)$", answer, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            if answer:
                return answer
    
    # 策略4: 如果response很长且没有"Answer:"标记，可能是纯rationale
    # 尝试从最后几行提取可能的答案（数字、选项字母等）
    if len(text) > 200:
        # 检查最后几行是否包含答案格式（数字、选项字母等）
        last_lines = lines[-5:] if len(lines) >= 5 else lines
        for line in reversed(last_lines):
            line = line.strip()
            # 匹配单个数字（可能是距离、深度等答案）
            if re.match(r'^-?\d+(\.\d+)?$', line):
                return line
            # 匹配选项字母（A/B/C/D）
            if re.match(r'^[A-D]$', line, re.IGNORECASE):
                return line.upper()
            # 匹配"Yes"或"No"
            if re.match(r'^(yes|no)$', line, re.IGNORECASE):
                return line.capitalize()
    
    # 策略5: 如果都找不到，且response很短，返回整个response
    # 如果response很长，可能是纯rationale，返回空字符串（表示提取失败）
    if len(text) < 200:
        return text
    else:
        # 长response且没有找到answer，可能是纯rationale，返回空
        return ""


def clean_response(text: str, question: str) -> str:
    """清理response，移除question重复等"""
    if not text:
        return text
    t = text.strip()
    q = (question or "").strip()
    lines = [line.rstrip() for line in t.splitlines()]
    if q and lines and lines[0].strip() == q:
        lines = lines[1:]
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower in ("assistant", "assistant:") or lower.startswith("assistant:"):
            lines = lines[i + 1 :]
            break
    return "\n".join(lines).strip()


def _resize_image(image, image_size: int):
    if not image_size or image_size <= 0:
        return image
    w, h = image.size
    if max(w, h) <= image_size:
        return image
    scale = image_size / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)


def infer_one_file_hf(
    in_jsonl: str,
    out_jsonl: str,
    model_path: str,
    lora_path: str,
    max_tokens: int,
    batch_size: int,
    image_size: int,
    use_scene_slot: bool,
    device: str = None,
    use_auto_device_map: bool = True,
    rank: int = 0,
    world_size: int = 1,
):
    print(f"[Infer] Loading data from {in_jsonl}")
    data = load_jsonl(in_jsonl)
    if lora_path and PeftModel is None:
        raise RuntimeError("peft is required to load LoRA weights in HF inference.")

    print("[Infer] Loading processor ...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    print("[Infer] Loading HF model ...")
    if device is not None and not use_auto_device_map:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=DTYPE,
            trust_remote_code=True,
            device_map={"": device},
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=DTYPE,
                trust_remote_code=True,
                device_map="auto",
            )
    if lora_path:
        print(f"[Infer] Loading LoRA from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path)
        print("[Infer] Merging LoRA weights ...")
        model = model.merge_and_unload()
    model.eval()

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        batch_images = []
        batch_prompts = []
        batch_meta = []
        for i, sample in enumerate(data):
            if world_size > 1 and (i % world_size) != rank:
                continue
            question, answer = build_qa(sample)
            scene_slot = get_scene_slot(sample) if use_scene_slot else None
            image_list = sample.get("image", [])
            if not question or not answer or not image_list:
                print(f"[Infer] Skipping sample {i} due to missing data")
                continue
            if use_scene_slot and scene_slot is None:
                print(f"[Infer] Skipping sample {i} due to missing scene slot")
                continue
            image_path = image_list[0]
            if not os.path.isfile(image_path):
                continue
            img = Image.open(image_path).convert("RGB")
            img = _resize_image(img, image_size)
            prompt = build_prompt(processor, question, scene_slot, use_scene_slot)
            batch_images.append(img)
            batch_prompts.append(prompt)
            batch_meta.append(
                {
                    "id": sample.get("id"),
                    "question": question,
                    "answer": answer,
                    "type": sample.get("type"),
                    "scene_slot": scene_slot if use_scene_slot else None,
                    "image": image_path,
                }
            )
            if len(batch_images) >= batch_size:
                _run_hf_batch(
                    model, processor, batch_images, batch_prompts, batch_meta, f_out, max_tokens
                )
                batch_images, batch_prompts, batch_meta = [], [], []
                print(f"[Infer] {i+1}/{len(data)}")

        if batch_images:
            _run_hf_batch(
                model, processor, batch_images, batch_prompts, batch_meta, f_out, max_tokens
            )

    print(f"[Infer] Saved predictions to {out_jsonl}")
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_hf_batch(model, processor, images, prompts, metas, f_out, max_tokens):
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    attention_mask = inputs.get("attention_mask")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    decoded = []
    if attention_mask is not None:
        input_lens = attention_mask.sum(dim=1).tolist()
        for seq, input_len in zip(output_ids, input_lens):
            gen_ids = seq[int(input_len):]
            decoded.append(
                processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            )
    else:
        decoded = processor.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
    for resp, meta in zip(decoded, metas):
        # 从response中提取answer部分
        question = meta.get("question", "")
        full_response = clean_response(resp.strip(), question)
        answer_only = extract_answer_from_response(full_response, question)
        
        # 如果提取失败（返回空字符串），说明可能是纯rationale
        # 在这种情况下，我们仍然保存完整response，但添加警告标记
        if not answer_only:
            # 如果response很长（>200字符），很可能是rationale，记录警告
            if len(full_response) > 200:
                print(f"[Warning] Failed to extract answer for sample {meta.get('id', 'unknown')}. "
                      f"Response length: {len(full_response)}. Using full response as fallback.")
            answer_only = full_response
        
        meta["response"] = answer_only
        f_out.write(json.dumps(meta, ensure_ascii=False) + "\n")


def merge_shards(base_path: str, world_size: int, remove_shards: bool = False):
    tmp_path = base_path + ".merge_tmp"
    with open(tmp_path, "w", encoding="utf-8") as f_out:
        for rank in range(world_size):
            shard = f"{base_path}.rank{rank}"
            if not os.path.isfile(shard):
                continue
            with open(shard, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f_out.write(line)
            if remove_shards:
                try:
                    os.remove(shard)
                except OSError:
                    pass
    os.replace(tmp_path, base_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/dsq/Models/Qwen2.5-VL-3B-Instruct",
        help="Base VLM path",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="LoRA adapter path (contains adapter_model.safetensors)",
    )
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--use-scene-slot",
        action="store_true",
        help="Include scene_slot in prompt and save it into pred.jsonl.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, distributed = init_distributed()
    if not distributed:
        raise RuntimeError("This script only supports hf_dp; please run with torchrun --nproc_per_node=8.")
    shard_out = f"{args.pred_jsonl}.rank{rank}"
    infer_one_file_hf(
        in_jsonl=args.val_jsonl,
        out_jsonl=shard_out,
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_scene_slot=args.use_scene_slot,
        device=f"cuda:{local_rank}" if torch.cuda.is_available() else None,
        use_auto_device_map=False,
        rank=rank,
        world_size=world_size,
    )
    safe_barrier(local_rank)
    if rank == 0:
        merge_shards(args.pred_jsonl, world_size, remove_shards=True)
        print(f"[Infer] Merged shards -> {args.pred_jsonl}")
        _sanitize_distributed_env()
        torch.cuda.empty_cache()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
