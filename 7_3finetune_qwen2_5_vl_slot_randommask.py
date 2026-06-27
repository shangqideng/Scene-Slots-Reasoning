"""
使用 image-question-answer 格式微调 Qwen2.5-VL-3B-Instruct。

输入：
- train jsonl (每行包含 image 和 conversations[human/gpt])

输出：
- 训练后的模型权重保存到 output_dir

示例：
torchrun --nproc_per_node=8 7finetune_qwen2_5_vl.py \
  --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.sam2.jsonl \
  --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
  --output-dir ./Models/3b_baseline_ft
  --lora
"""
import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        TrainingArguments,
        Trainer,
    )
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception:  # pragma: no cover
        Qwen2_5_VLForConditionalGeneration = None
except Exception as exc:  # pragma: no cover
    AutoProcessor = None
    AutoModelForVision2Seq = None
    TrainingArguments = None
    Trainer = None
    Qwen2_5_VLForConditionalGeneration = None
    _TF_IMPORT_ERROR = exc

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None


def get_qa(sample):
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    if question is None or answer is None:
        return None, None
    return str(question).strip(), str(answer).strip()


def get_image_path(sample):
    image_list = sample.get("image", [])
    if image_list:
        return image_list[0]
    return None


def get_scene_slot(sample):
    return sample.get("scene_slot")


class JsonlIndexDataset(Dataset):
    def __init__(self, path, max_samples=None, skip_missing_image=True):
        self.path = path
        self.offsets = []
        self.max_samples = max_samples
        self.skip_missing_image = skip_missing_image
        self._file = None
        self._build_index()

    def _build_index(self):
        with open(self.path, "r", encoding="utf-8") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q, a = get_qa(obj)
                img = get_image_path(obj)
                slot = get_scene_slot(obj)
                if q is None or a is None:
                    continue
                if self.skip_missing_image and (not img or not os.path.isfile(img)):
                    continue
                if slot is None:
                    continue
                self.offsets.append(pos)
                if self.max_samples and len(self.offsets) >= self.max_samples:
                    break

    def __len__(self):
        return len(self.offsets)

    def _get_file(self):
        if self._file is None:
            self._file = open(self.path, "r", encoding="utf-8")
        return self._file

    def __getitem__(self, idx):
        f = self._get_file()
        f.seek(self.offsets[idx])
        line = f.readline()
        obj = json.loads(line)
        q, a = get_qa(obj)
        img = get_image_path(obj)
        slot = get_scene_slot(obj)
        return {
           "id": obj.get("id"),
            "image": img,
            "question": q,
            "answer": a, 
            "scene_slot": slot,
        }


class QwenVLDataCollator:
    def __init__(self, processor, image_size=None, max_text_tokens=None, random_mask_ratio=0.3):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self._printed_debug = False
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.random_mask_ratio = max(0.0, min(1.0, float(random_mask_ratio)))

    def _resize_image(self, image):
        if not self.image_size:
            return image
        max_side = int(self.image_size)
        if max_side <= 0:
            return image
        w, h = image.size
        if max(w, h) <= max_side:
            return image
        scale = max_side / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return image.resize((new_w, new_h), resample=Image.BICUBIC)

    def _apply_random_mask(self, image):
        ratio = self.random_mask_ratio
        if ratio <= 0.0:
            return image
        arr = np.array(image)
        if arr.ndim != 3:
            return image
        h, w, _ = arr.shape
        if h <= 0 or w <= 0:
            return image
        mask = np.random.random((h, w)) < ratio
        arr[mask] = 0
        return Image.fromarray(arr)

    def _build_prompt(self, question, scene_slot):
        slot_text = json.dumps(scene_slot, ensure_ascii=False)
        text = (
            f"{question}\n\n"
            "Scene slot (coords: x-right, y-down, z-forward):\n"
            f"{slot_text}"
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
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def __call__(self, batch):
        images = []
        prompts = []
        full_texts = []
        first_item = None
        for item in batch:
            img = Image.open(item["image"]).convert("RGB")
            img = self._resize_image(img)
            img = self._apply_random_mask(img)
            question = item["question"]
            answer = item["answer"]
            scene_slot = item.get("scene_slot")
            prompt = self._build_prompt(question, scene_slot)
            eos = self.tokenizer.eos_token or ""
            full = prompt + answer + eos # eos就是'<|im_end|>'
            if first_item is None:
                first_item = {
                    "question": question,
                    "answer": answer,
                    "scene_slot": scene_slot,
                    "prompt": prompt,
                    "full": full,
                }
            images.append(img)
            prompts.append(prompt)
            full_texts.append(full)

        full_inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=bool(self.max_text_tokens),
            max_length=self.max_text_tokens,
        )
        prompt_inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=bool(self.max_text_tokens),
            max_length=self.max_text_tokens,
        )

        input_ids = full_inputs["input_ids"]
        labels = input_ids.clone()
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
        # prompt和padding位置的label设为-100
        for i in range(labels.size(0)):
            labels[i, : prompt_lens[i]] = -100
            labels[i, full_inputs["attention_mask"][i] == 0] = -100
        # 不对 <|im_end|> / eos 计算loss
        eos_id = self.tokenizer.eos_token_id
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is not None and eos_id >= 0:
            labels[input_ids == eos_id] = -100
        if isinstance(im_end_id, int) and im_end_id >= 0:
            labels[input_ids == im_end_id] = -100

        full_inputs["labels"] = labels
        # if not self._printed_debug and first_item is not None:
        #     print("=== Debug: First Sample ===")
        #     print("eos_token:", repr(self.tokenizer.eos_token))
        #     print("question:", first_item["question"])
        #     print("answer:", first_item["answer"])
        #     print("prompt:", first_item["prompt"])
        #     print("full:", first_item["full"])
        #     print("labels:", labels[0].tolist())
        #     print("=== End Debug ===")
        #     self._printed_debug = True
        return full_inputs


class LossRecorderTrainer(Trainer):
    def __init__(self, *args, loss_recorder=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_recorder = loss_recorder

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.loss_recorder is not None and self.is_world_process_zero():
            try:
                self.loss_recorder.append(float(loss.detach().float().item()))
            except Exception:
                self.loss_recorder.append(float(loss))
        return loss


def build_model_and_processor(model_path, use_bf16=True):
    if AutoProcessor is None:
        raise RuntimeError(f"transformers import failed: {_TF_IMPORT_ERROR}")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    if Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
    return model, processor


def build_checkpoint_kwargs():
    return {"use_reentrant": True}


def enable_input_require_grads_for_model(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return
    base_getter = getattr(model, "get_base_model", None)
    if callable(base_getter):
        base_model = base_getter()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()


def load_fsdp_config(path):
    if not path:
        return None
    if not os.path.isfile(path):
        raise RuntimeError(f"fsdp config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with IQA data")
    parser.add_argument("--train-jsonl", required=True, help="Train jsonl path")
    parser.add_argument(
        "--model-path",
        default="/data/dsq/Models/Qwen2.5-VL-3B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--output-dir",
        default="./Models/3b_baseline_ft",
        help="Output model directory",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--per-device-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Resize image so the longest side equals this value",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=None,
        help="Optional max text tokens for prompt/answer; truncate if set.",
    )
    parser.add_argument(
        "--random-mask-ratio",
        type=float,
        default=0.2,
        help="Randomly mask this ratio of image pixels (0 to disable).",
    )
    parser.add_argument("--lora", action="store_true", help="Enable LoRA finetuning")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--fsdp",
        default="full_shard auto_wrap",
        help="Enable FSDP (set to 'none' to disable)",
    )
    parser.add_argument(
        "--fsdp-config",
        default="./fsdp_config.json",
        help="Path to FSDP config json",
    )
    parser.add_argument(
        "--no-bf16", action="store_true", help="Disable bf16 training"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--skip-missing-image",
        action="store_true",
        help="Skip samples with missing image",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not os.path.isfile(args.train_jsonl):
        raise RuntimeError(f"train jsonl not found: {args.train_jsonl}")
    model, processor = build_model_and_processor(
        args.model_path,
        use_bf16=not args.no_bf16,
    )
    if args.lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft is required for LoRA (pip install peft)")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
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
        model = get_peft_model(model, lora_config)
        # model.get_input_embeddings().weight.requires_grad_(True)
    # 训练时禁用 KV cache，避免注意力维度不匹配
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "generation_config") and hasattr(
        model.generation_config, "use_cache"
    ):
        model.generation_config.use_cache = False
    if args.gradient_checkpointing:
        # Use reentrant checkpointing to avoid dtype-mismatch errors during recompute.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=build_checkpoint_kwargs()
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        enable_input_require_grads_for_model(model)

    dataset = JsonlIndexDataset(
        args.train_jsonl,
        max_samples=args.max_samples,
        skip_missing_image=args.skip_missing_image,
    )
    collator = QwenVLDataCollator(
        processor,
        image_size=args.image_size,
        max_text_tokens=args.max_text_tokens,
        random_mask_ratio=args.random_mask_ratio,
    )

    fsdp = args.fsdp
    if fsdp and str(fsdp).lower() in ("none", "false", "0"):
        fsdp = None
    fsdp_config = load_fsdp_config(args.fsdp_config) if fsdp else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=not args.no_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        fsdp=fsdp,
        fsdp_config=fsdp_config,
        seed=args.seed,
        report_to=[],
    )

    losses = []
    trainer = LossRecorderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        loss_recorder=losses,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "losses.npy"), np.array(losses))
    print(f"Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
