'''
torchrun --nproc_per_node=8 8infer_and_score_vllm.py \
  --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
  --lora-path /home/lxp/Ground_reasoning/Models/3b_baseline_ft_lr1e-5 \
  --val-jsonl /data/dsq/ScanNet/qa_jsonl/val.sam2.jsonl \
  --pred-jsonl /data/dsq/ScanNet/qa_jsonl/infer/3b_baseline_ft_lr1e-511/val.pred.jsonl

仅用于 hf_dp 推理（torchrun 8 卡分片）。
'''
import os
import json
import gc
import argparse

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
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    return question, answer


def get_scene_slot(sample):
    return sample.get("scene_slot")


def build_prompt(processor, question: str, scene_slot=None, use_scene_slot: bool = False) -> str:
    if use_scene_slot:
        slot_text = json.dumps(scene_slot, ensure_ascii=False)
        text = (
            f"{question}\n\n"
            "Scene slot (coords: x-right, y-down, z-forward):\n"
            f"{slot_text}"
        )
    else:
        text = question
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


def clean_response(text: str, question: str) -> str:
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


def _load_lora_rank(lora_path: str) -> int:
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            r = int(cfg.get("r", 0))
            if r > 0:
                return r
        except Exception:
            pass
    return 64


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
        model = PeftModel.from_pretrained(model, lora_path)
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
        meta["response"] = clean_response(resp.strip(), meta.get("question"))
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
        help="Finetuned VLM path (merged) or base model when using LoRA",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA adapter path (contains adapter_model.safetensors)",
    )
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
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
