#!/bin/bash
# 推理训练好的rationale SFT模型
# 使用8卡分布式推理，只提取answer部分用于评分

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 模型路径
# 基座模型：与训练脚本中一致，使用原始Qwen2.5-VL-3B-Instruct
MODEL_PATH="/data/dsq/Models/Qwen2.5-VL-3B-Instruct"
# 使用训练脚本 RL/train_rationale_sft.sh 得到的rationale SFT权重
LORA_PATH="/data/dsq/Ground_reasoning/Models_bak/rationale_sft/2e-4_short_rmtag/v0-20260305-032555/checkpoint-1326"

# 数据路径
VAL_JSONL="/data/dsq/ScanNet/qa_jsonl/val.question_scene_slot_correct_rename_add2dmask_addnoise.jsonl"
PRED_JSONL="/data/dsq/ScanNet/qa_jsonl/infer/against_noisy/add_noisy_testcomputer/val.pred.jsonl"

# 推理参数
BATCH_SIZE=32
MAX_TOKENS=4096
IMAGE_SIZE=256

# 执行推理
# 使用checkpoint-384作为基座模型，训练后的checkpoint作为lora权重
torchrun --nproc_per_node=8 RL/infer_rationale_sft.py \
  --model-path ${MODEL_PATH} \
  --lora-path ${LORA_PATH} \
  --val-jsonl ${VAL_JSONL} \
  --pred-jsonl ${PRED_JSONL} \
  --batch-size ${BATCH_SIZE} \
  --max-tokens ${MAX_TOKENS} \
  --image-size ${IMAGE_SIZE} \
  --use-scene-slot

echo "Inference completed. Predictions saved to: ${PRED_JSONL}"
echo "Next step: Run scoring with 9score_only_vllm.py"