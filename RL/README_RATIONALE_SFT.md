# Rationale SFT训练指南

使用ms-swift框架对Qwen2.5-VL-3B-Instruct进行SFT冷启动训练，训练模型能够根据Q（question）、I（image）、slot（scene_slot）同时生成rationale和answer。

## 训练策略

每个样本随机分配为三种训练模式之一（1:1:1比例）：
- **rationale_only**: 只对rationale部分计算loss（loss=true），answer部分不计算loss（loss=false）
- **answer_only**: 只对answer部分计算loss（loss=true），rationale部分不计算loss（loss=false）
- **both**: 对rationale和answer都计算loss（loss=true）

**关键实现**：使用ms-swift>=3.8的`loss`字段功能来控制label mask，无需自定义collator。每个assistant消息包含`loss`字段，框架会自动处理。

这样可以防止模型只学会单一能力，确保模型既能生成rationale，也能生成answer。

## 使用步骤

### 1. 转换数据格式

首先将rationale jsonl转换为ms-swift训练格式：

```bash
python RL/convert_rationale_to_swift.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl \
  --seed 42
```

转换后的数据格式：
- 输入：Q（question）+ I（image）+ slot（scene_slot）
- 输出：rationale + "\n\nAnswer: " + answer
- 每个样本包含`training_mode`字段，标记为"rationale_only"、"answer_only"或"both"

### 2. 训练模型

使用ms-swift进行训练：

```bash
# 使用训练脚本
bash RL/train_rationale_sft.sh

# 或直接使用swift命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
  --model /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
  --dataset /data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl \
  --tuner_type lora \
  --lora_rank 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --output_dir ./output/rationale_sft \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --save_steps 500 \
  --save_total_limit 2 \
  --logging_steps 50 \
  --bf16 true \
  --dataloader_num_workers 8 \
  --max_length 4096 \
  --remove_unused_columns false \
  --report_to none
```

### 3. 训练完成

训练完成后，模型会保存在`${OUTPUT_DIR}`目录下，可以直接用于后续的GRPO训练。

## 数据格式说明

### 输入格式（转换后，ms-swift标准messages格式）

```json
{
  "id": "scene0706_00_72",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {
          "type": "text",
          "text": "Question here\n\nScene slot (coords: x-right, y-down, z-forward):\n{scene_slot_json}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Rationale text here",
      "loss": true
    },
    {
      "role": "assistant",
      "content": "Answer: answer text",
      "loss": true
    }
  ],
  "images": ["/path/to/image.jpg"],
  "training_mode": "both",
  "scene_slot": [...]
}
```

**关键点**：
- 使用`messages`格式（ms-swift标准格式）
- rationale和answer分成两个assistant消息
- 每个assistant消息包含`loss`字段，控制是否计算loss
- `images`字段存储图片路径

### Training Mode说明

- **rationale_only**: 训练时只对rationale部分计算loss
- **answer_only**: 训练时只对answer部分计算loss
- **both**: 训练时对rationale和answer都计算loss

## 训练参数说明

参考`7finetune_qwen2_5_vl_slot.py`的训练配置：
- LoRA rank: 32
- LoRA alpha: 16
- Learning rate: 1e-5
- Batch size: 1 per device
- Gradient accumulation: 8
- Epochs: 2
- Max length: 4096

## 注意事项

1. **Loss字段功能**：需要ms-swift>=3.8版本，该版本支持通过`loss`字段控制是否计算损失。这是框架原生功能，比自定义collator更可靠。

2. **数据格式**：使用ms-swift标准`messages`格式，rationale和answer分成两个assistant消息，分别设置`loss`字段。

3. **验证数据**：转换后建议检查数据格式，确认每个assistant消息都有`loss`字段：
   ```bash
   python -c "
   import json
   with open('/data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl', 'r') as f:
       sample = json.loads(f.readline())
       for msg in sample['messages']:
           if msg['role'] == 'assistant':
               print(f\"Loss: {msg.get('loss', 'None')}, Content: {msg['content'][:50]}...\")
   "
   ```

3. **数据检查**：训练前建议检查转换后的数据格式：
   ```bash
   python -c "
   import json
   with open('/data/dsq/ScanNet/qa_jsonl/all.question.rl.swift.jsonl', 'r') as f:
       sample = json.loads(f.readline())
       print(f\"Training mode: {sample.get('training_mode')}\")
       for i, msg in enumerate(sample['messages']):
           if msg['role'] == 'assistant':
               print(f\"  Assistant {i}: loss={msg.get('loss')}, content_len={len(msg['content'])}\")
   "
   ```

## 下一步：GRPO训练

完成SFT冷启动后，可以使用训练好的模型进行GRPO训练，进一步提升模型性能。
