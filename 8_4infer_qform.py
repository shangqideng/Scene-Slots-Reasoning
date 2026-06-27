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
from torch import nn
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import PreTrainedModel

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


DTYPE = torch.bfloat16

SLOT_TOKEN = "<|scene_slot|>"


class SlotQFormer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_size))
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.image_proj = nn.Linear(3, hidden_size)

    def forward(self, source_embeds, source_mask=None):
        if source_embeds is None:
            return None
        bsz = source_embeds.size(0)
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        key_padding_mask = None
        if source_mask is not None:
            key_padding_mask = ~source_mask.bool()
        out, _ = self.attn(query, source_embeds, source_embeds, key_padding_mask=key_padding_mask)
        return self.ln(out)

    def forward_image(self, pixel_values):
        if pixel_values is None:
            return None
        pooled = pixel_values.mean(dim=(2, 3))
        source = self.image_proj(pooled).unsqueeze(1)
        return self.forward(source)


class QwenWithSlotQFormer(nn.Module):
    def __init__(self, base_model, slot_token_id: int, qformer_source: str = "scene_slot"):
        super().__init__()
        self.base_model = base_model
        self.slot_token_id = slot_token_id
        self.qformer_source = qformer_source
        self.embed = base_model.get_input_embeddings()
        hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = self.embed.weight.shape[1]
        self.qformer = SlotQFormer(hidden_size)
        self.config = base_model.config

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, **kwargs):

        # 1️⃣ 取出自定义参数
        slot_input_ids = kwargs.pop("slot_input_ids", None)
        slot_attention_mask = kwargs.pop("slot_attention_mask", None)
        question_input_ids = kwargs.pop("question_input_ids", None)
        question_attention_mask = kwargs.pop("question_attention_mask", None)

        # 2️⃣ 保存到临时属性，供 forward 使用
        self._slot_input_ids = slot_input_ids
        self._slot_attention_mask = slot_attention_mask
        self._question_input_ids = question_input_ids
        self._question_attention_mask = question_attention_mask

        # 3️⃣ monkey patch forward
        original_forward = self.base_model.forward
        self.base_model.forward = self.forward

        try:
            outputs = self.base_model.generate(**kwargs)
        finally:
            self.base_model.forward = original_forward

            # 清理缓存
            self._slot_input_ids = None
            self._slot_attention_mask = None
            self._question_input_ids = None
            self._question_attention_mask = None

        return outputs

    
    def _compute_slot_vec(
        self,
        slot_input_ids,
        slot_attention_mask,
        question_input_ids,
        question_attention_mask,
        pixel_values,
    ):
        if self.qformer_source == "scene_slot":
            if slot_input_ids is None:
                return None
            source_embeds = self.embed(slot_input_ids)
            return self.qformer(source_embeds, slot_attention_mask)
        if self.qformer_source == "question":
            if question_input_ids is None:
                return None
            source_embeds = self.embed(question_input_ids)
            return self.qformer(source_embeds, question_attention_mask)
        if self.qformer_source == "image":
            return self.qformer.forward_image(pixel_values)
        return None

    def forward(self, **kwargs):
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.get("attention_mask")
        pixel_values = kwargs.get("pixel_values")
        slot_input_ids = kwargs.pop("slot_input_ids", None)
        slot_attention_mask = kwargs.pop("slot_attention_mask", None)
        question_input_ids = kwargs.pop("question_input_ids", None)
        question_attention_mask = kwargs.pop("question_attention_mask", None)
        if input_ids is None:
            return self.base_model(**kwargs)

        inputs_embeds = self.embed(input_ids)
        slot_vec = self._compute_slot_vec(
            slot_input_ids,
            slot_attention_mask,
            question_input_ids,
            question_attention_mask,
            pixel_values,
        )
        if slot_vec is not None:
            slot_mask = (input_ids == self.slot_token_id).unsqueeze(-1)
            if slot_mask.any():
                slot_vec = slot_vec[:, 0:1, :]
                inputs_embeds = torch.where(slot_mask, slot_vec.expand_as(inputs_embeds), inputs_embeds)

        kwargs["inputs_embeds"] = inputs_embeds
        return self.base_model(**kwargs)


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


def build_prompt(processor, question: str, use_scene_slot: bool = False) -> str:
    if use_scene_slot:
        text = (
            f"{question}\n\n"
            "Scene slot (coords: x-right, y-down, z-forward):\n"
            f"{SLOT_TOKEN}"
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
    qformer_source: str,
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
        model_path, trust_remote_code=True, use_fast=False, local_files_only=True
    )

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [SLOT_TOKEN]}
        )

    print("[Infer] Loading base model ...")

    if device is not None and not use_auto_device_map:
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=DTYPE,
            trust_remote_code=True,
            device_map={"": device},
        )
    else:
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=DTYPE,
            trust_remote_code=True,
            device_map="auto",
        )

    # resize embedding（必须在包装前）
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        base_model.resize_token_embeddings(len(processor.tokenizer))
        slot_token_id = processor.tokenizer.convert_tokens_to_ids(SLOT_TOKEN)
    else:
        slot_token_id = None

    if slot_token_id is None or slot_token_id < 0:
        raise RuntimeError("Slot token id not found.")

    # 如果训练时开了 LoRA，推理也必须构造 LoRA 结构
    if lora_path:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        base_model = get_peft_model(base_model, lora_config)

    # 构造 QFormer wrapper
    model = QwenWithSlotQFormer(
        base_model,
        slot_token_id=slot_token_id,
        qformer_source=qformer_source,
    )

    # ======= 关键：加载完整 state_dict =======
    import safetensors.torch

    model_file = os.path.join(lora_path, "model.safetensors")

    print(f"[Infer] Loading trained weights from {model_file}")
    state_dict = safetensors.torch.load_file(model_file)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

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
            prompt = build_prompt(processor, question, use_scene_slot)
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
                    model,
                    processor,
                    batch_images,
                    batch_prompts,
                    batch_meta,
                    f_out,
                    max_tokens,
                    use_scene_slot,
                )
                batch_images, batch_prompts, batch_meta = [], [], []
                print(f"[Infer] {i+1}/{len(data)}")

        if batch_images:
            _run_hf_batch(
                model,
                processor,
                batch_images,
                batch_prompts,
                batch_meta,
                f_out,
                max_tokens,
                use_scene_slot,
            )

    print(f"[Infer] Saved predictions to {out_jsonl}")
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_hf_batch(model, processor, images, prompts, metas, f_out, max_tokens, use_scene_slot):
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    if use_scene_slot and hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        slot_texts = [json.dumps(m.get("scene_slot"), ensure_ascii=False) for m in metas]
        question_texts = [m.get("question") or "" for m in metas]
        slot_inputs = processor.tokenizer(
            slot_texts, return_tensors="pt", padding=True, truncation=False
        )
        question_inputs = processor.tokenizer(
            question_texts, return_tensors="pt", padding=True, truncation=False
        )
        inputs["slot_input_ids"] = slot_inputs["input_ids"]
        inputs["slot_attention_mask"] = slot_inputs["attention_mask"]
        inputs["question_input_ids"] = question_inputs["input_ids"]
        inputs["question_attention_mask"] = question_inputs["attention_mask"]
    attention_mask = inputs.get("attention_mask")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(next(model.parameters()).device)
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
    parser.add_argument(
        "--slot-qformer-source",
        choices=["scene_slot", "question", "image"],
        default="scene_slot",
        help="Source for Q-Former slot vector: scene_slot/question/image.",
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
        qformer_source=args.slot_qformer_source,
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
