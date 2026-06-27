"""
将rationale jsonl转换为ms-swift训练格式。

每个样本随机分配为三种训练模式之一（1:1:1）：
- rationale_only: 只训练生成rationale
- answer_only: 只训练生成answer
- both: 同时训练生成rationale和answer

使用方法：
python RL/convert_rationale_to_swift.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl \
  --max-rationale-length 6000
  
Conversion completed:
  Total input samples: 27112
  Converted samples: 27112
  Training mode distribution:
    rationale_only: 9047 (33.4%)
    answer_only: 8988 (33.2%)
    both: 9077 (33.5%)
  Output file: /data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl

在模版里就要有<thinking>和<answer>标签，这样才能SFT学到模板化！！！
{
  "messages": [
    {"role": "user", "content": "..." },
    {"role": "assistant", "content": "<thinking>这段不监督</thinking>", "loss": false},
    {"role": "assistant", "content": "<answer>Yes</answer>", "loss": true}
  ]
}
或者：
{
  "id": "mm_0001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/data/dsq/ScanNet/images/xxx.jpg"},
        {"type": "text", "text": "图中红色物体在蓝色物体的左边吗？请给出理由，并最终回答Yes/No。"}
      ]
    },
    {
      "role": "assistant",
      "content": "<thinking>...</thinking><answer>Yes</answer>"
    }
  ]
}
"""
import argparse
import json
import os
import random


def get_qa(sample):
    """从样本中提取问题和答案"""
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


def get_scene_slot(sample):
    """从样本中提取scene_slot"""
    return sample.get("scene_slot")


def convert_to_swift_format(sample, training_mode, max_rationale_length=None):
    """
    转换为ms-swift格式。
    
    training_mode: 'rationale_only', 'answer_only', 'both'
    max_rationale_length: 如果设置，会截断rationale文本（按字符数）
    """
    question, answer = get_qa(sample)
    rationale = sample.get("rationale", "")
    scene_slot = get_scene_slot(sample)
    image = sample.get("image", [])
    
    if not question or not answer:
        return None
    
    # 如果设置了最大长度，截断rationale（保留前面的部分）
    if max_rationale_length and rationale and len(rationale) > max_rationale_length:
        rationale = rationale[:max_rationale_length] + "..."
    
    # 构建scene_slot文本
    slot_text = json.dumps(scene_slot, ensure_ascii=False) if scene_slot else "{}"
    
    # 构建输入文本（包含question和scene_slot）
    input_text = f"{question}\n\nScene slot (coords: x-right, y-down, z-forward):\n{slot_text}"
    
    # 根据training_mode设置loss字段，使用ms-swift的loss字段控制label mask
    # 将rationale和answer分成两个assistant消息，分别控制loss
    # 这样可以根据training_mode精确控制哪些部分计算loss
    
    # 构建messages格式（ms-swift标准格式）
    messages = []
    
    # User消息
    # 确保content字段格式一致：如果有图片，使用数组格式；否则也使用数组格式（包含text）
    if image:
        # 多模态：使用content数组
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": input_text}
            ]
        })
    else:
        # 纯文本：也使用数组格式以保持一致性（ms-swift要求）
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": input_text}
            ]
        })
    
    # Assistant消息1: rationale部分
    # 注意：为了保持content字段类型一致（都是数组），assistant消息的content也使用数组格式
    rationale_loss = training_mode in ["rationale_only", "both"]
    if rationale:  # 只有当rationale存在时才添加
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<think>{rationale}</think>"}
            ],
            "loss": rationale_loss  # ms-swift>=3.8支持loss字段控制是否计算loss
        })
    
    # Assistant消息2: answer部分
    answer_loss = training_mode in ["answer_only", "both"]
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": f"<answer>{answer}</answer>"}
        ],
        "loss": answer_loss  # ms-swift>=3.8支持loss字段控制是否计算loss
    })
    
    # 构建ms-swift标准格式
    swift_sample = {
        "messages": messages,
    }
    
    # 添加images字段（多模态数据，只有当有图片时才添加）
    if image:
        swift_sample["images"] = image
    
    # 保留原始字段用于调试（ms-swift会自动忽略未识别的字段）
    swift_sample["id"] = sample.get("id", "")
    swift_sample["training_mode"] = training_mode
    swift_sample["scene_slot"] = scene_slot
    
    return swift_sample


def convert_jsonl(input_path, output_path, seed=42, max_rationale_length=None):
    """转换jsonl文件"""
    random.seed(seed)
    
    # 始终使用both模式（两个assistant的loss都为True）
    training_modes = ["both"]
    mode_counts = {"rationale_only": 0, "answer_only": 0, "both": 0}
    
    total_samples = 0
    converted_samples = 0
    truncated_count = 0
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            
            try:
                sample = json.loads(line)
                total_samples += 1
                
                # 训练模式固定为both
                training_mode = random.choice(training_modes)
                
                # 检查rationale是否会被截断
                original_rationale = sample.get("rationale", "")
                if max_rationale_length and original_rationale and len(original_rationale) > max_rationale_length:
                    truncated_count += 1
                
                # 转换格式
                swift_sample = convert_to_swift_format(sample, training_mode, max_rationale_length)
                
                if swift_sample:
                    f_out.write(json.dumps(swift_sample, ensure_ascii=False) + "\n")
                    converted_samples += 1
                    mode_counts[training_mode] += 1
                    
                    if converted_samples % 1000 == 0:
                        print(f"Converted {converted_samples} samples...")
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"\nConversion completed:")
    print(f"  Total input samples: {total_samples}")
    print(f"  Converted samples: {converted_samples}")
    if truncated_count > 0:
        print(f"  Truncated samples: {truncated_count} (rationale length > {max_rationale_length})")
    print(f"  Training mode distribution:")
    if converted_samples > 0:
        print(f"    rationale_only: {mode_counts['rationale_only']} ({mode_counts['rationale_only']/converted_samples*100:.1f}%)")
        print(f"    answer_only: {mode_counts['answer_only']} ({mode_counts['answer_only']/converted_samples*100:.1f}%)")
        print(f"    both: {mode_counts['both']} ({mode_counts['both']/converted_samples*100:.1f}%)")
    print(f"  Output file: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert rationale jsonl to ms-swift format")
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Input jsonl file path"
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output jsonl file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training mode assignment"
    )
    parser.add_argument(
        "--max-rationale-length",
        type=int,
        default=None,
        help="Maximum length for rationale text (in characters). If set, longer rationales will be truncated."
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    
    if not os.path.isfile(args.input_jsonl):
        raise RuntimeError(f"Input file not found: {args.input_jsonl}")
    
    convert_jsonl(args.input_jsonl, args.output_jsonl, seed=args.seed, max_rationale_length=args.max_rationale_length)


if __name__ == "__main__":
    main()
