#!/bin/bash
# 推理训练好的 rationale GRPO 模型
# 使用8卡分布式推理，只提取answer部分用于评分

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 模型路径（请根据当前 GRPO 版本调整）
MODEL_PATH="/data/dsq/Models/Qwen2.5-VL-3B-Instruct"
LORA_PATH="/home/lxp/Ground_reasoning/Models/rationale_grpo/2e-4/"

# 数据路径（与 SFT 推理相同，便于对比评分）
VAL_JSONL="/data/dsq/ScanNet/qa_jsonl/val.question_scene_slot_correct_rename_add2dmask.jsonl"
PRED_JSONL="/data/dsq/ScanNet/qa_jsonl/infer/rationale_grpo/2e-4/val.pred.jsonl"

# 推理参数（适当放大 max_tokens，以容纳 <think> + <answer>）
BATCH_SIZE=4
MAX_TOKENS=4096
IMAGE_SIZE=256

# 执行推理
torchrun --nproc_per_node=8 RL/infer_rationale_grpo.py \
  --model-path ${MODEL_PATH} \
  --lora-path ${LORA_PATH} \
  --val-jsonl ${VAL_JSONL} \
  --pred-jsonl ${PRED_JSONL} \
  --batch-size ${BATCH_SIZE} \
  --max-tokens ${MAX_TOKENS} \
  --image-size ${IMAGE_SIZE} \
  --use-scene-slot

echo "GRPO inference completed. Predictions saved to: ${PRED_JSONL}"
echo "Next step: Run scoring with 9score_only_vllm.py"

