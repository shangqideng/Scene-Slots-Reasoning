import json
import os
from transformers import AutoProcessor

jsonl = "/data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.jsonl"
model_path = "/data/dsq/Models/Qwen2.5-VL-3B-Instruct"

print(f"Loading processor from {model_path} ...", flush=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer = processor.tokenizer
eos = tokenizer.eos_token or ""

print(f"Reading {jsonl} ...", flush=True)

def get_qa(sample):
    q = None
    a = None
    for item in sample.get("conversations", []):
        if item.get("from") == "human" and q is None:
            q = item.get("value")
        elif item.get("from") == "gpt" and a is None:
            a = item.get("value")
    if q is None or a is None:
        return None, None
    return str(q).strip(), str(a).strip()

def build_prompt(question, scene_slot):
    slot_text = json.dumps(scene_slot, ensure_ascii=False)
    text = f"{question}\n\nScene slot:\n{slot_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

lens = []
skipped = 0
with open(jsonl, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, a = get_qa(sample)
        if q is None or a is None:
            skipped += 1
            continue
        prompt = build_prompt(q, sample.get("scene_slot"))
        full = prompt + str(a).strip() + eos
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        lens.append((len(ids), sample.get("id")))

if not lens:
    print("No valid samples found.", flush=True)
    raise SystemExit(0)

lens.sort(key=lambda x: x[0])
lengths = [l for l, _ in lens]

def percentile(p):
    k = int(round((p / 100) * (len(lengths) - 1)))
    return lengths[k]

print(f"Total samples: {len(lengths)}, skipped: {skipped}", flush=True)
print(f"min={lengths[0]} max={lengths[-1]} mean={sum(lengths)/len(lengths):.1f}", flush=True)
for p in [50, 90, 95, 99, 99.5]:
    print(f"p{p}={percentile(p)}", flush=True)

print("Top 20 longest:", flush=True)
for l, sid in lens[-20:][::-1]:
    print(l, sid, flush=True)