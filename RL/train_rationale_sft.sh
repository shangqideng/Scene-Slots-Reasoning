#!/bin/bash
# 使用ms-swift框架基于已有LoRA checkpoint继续训练
# 基座模型：Qwen2.5-VL-3B-Instruct
# 已有LoRA checkpoint：checkpoint-384（从基座模型通过LoRA训练得到）
# 
# 训练目标：输入Q、I、slot，输出rationale和answer
# 训练策略：根据training_mode随机mask label（1:1:1）
# - rationale_only: 只训练生成rationale
# - answer_only: 只训练生成answer
# - both: 同时训练生成rationale和answer

# 设置GPU和CUDA内存分配
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# 模型和数据路径
# 基座模型：原始Qwen2.5-VL-3B-Instruct
BASE_MODEL_PATH="/data/dsq/Models/Qwen2.5-VL-3B-Instruct"
# 已有的LoRA checkpoint：基于基座模型训练得到的checkpoint-384
RESUME_CHECKPOINT="/home/lxp/Ground_reasoning/Models/question_scene_slot/3b_baseline_ft_lr2e-5_img256_correct_rename_add2dmask"
# /data/dsq/ScanNet/qa_jsonl/train.question.rl.swift.short.jsonl
# 短rationale都true且删除<>并且2e-4的sft
# DATASET_PATH="/data/dsq/ScanNet/qa_jsonl/train.question.rl.swift.short.rmtag.jsonl"
# 短rationale都tru并且2e-4的sft
# DATASET_PATH="/data/dsq/ScanNet/qa_jsonl/train.question.rl.swift.short.jsonl"


# 用ds并重写prompt生成的rationale
DATASET1_PATH="/data/dsq/ScanNet/qa_jsonl/train.question.rationale.modified_ds.jsonl"
##############################################################
OUTPUT1_DIR="./Models/rationale_sft/2e-4_modified_ds_max4096_base_slotsft"
swift sft \
  --model ${BASE_MODEL_PATH} \
  --model_type qwen2_5_vl \
  --adapters ${RESUME_CHECKPOINT} \
  --dataset ${DATASET1_PATH} \
  --train_type lora \
  --lora_rank 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --output_dir ${OUTPUT1_DIR} \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --logging_steps 50 \
  --bf16 true \
  --dataloader_num_workers 8 \
  --dataset_num_proc 4 \
  --max_length 4096 \
  --system "You are a 3D spatial understanding expert." \
  --remove_unused_columns false \
  --report_to none
echo "Training completed. Model saved to: ${OUTPUT1_DIR}"


# # 用ds并重写prompt并过滤掉不一致的纯干净rationale
# DATASET2_PATH="/data/dsq/ScanNet/qa_jsonl/train.question.rationale.modified_ds.filtered_by_llm.jsonl"
# ##############################################################
# OUTPUT2_DIR="./Models/rationale_sft/2e-4_modified_ds_filtered_by_llm_max4096"
# swift sft \
#   --model ${BASE_MODEL_PATH} \
#   --model_type qwen2_5_vl \
#   --adapters ${RESUME_CHECKPOINT} \
#   --dataset ${DATASET2_PATH} \
#   --train_type lora \
#   --lora_rank 32 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --target_modules all-linear \
#   --output_dir ${OUTPUT2_DIR} \
#   --num_train_epochs 2 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 2e-4 \
#   --weight_decay 0.0 \
#   --warmup_ratio 0.1 \
#   --save_steps 1000 \
#   --save_total_limit 2 \
#   --logging_steps 50 \
#   --bf16 true \
#   --dataloader_num_workers 8 \
#   --dataset_num_proc 4 \
#   --max_length 4096 \
#   --system "You are a 3D spatial understanding expert." \
#   --remove_unused_columns false \
#   --report_to none
# echo "Training completed. Model saved to: ${OUTPUT2_DIR}"