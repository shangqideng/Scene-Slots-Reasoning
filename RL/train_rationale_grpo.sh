#!/bin/bash
# 使用 ms-swift 对 Qwen2.5-VL-3B-Instruct + 冷启动 LoRA 进行 GRPO 训练
# 目标任务: distance_infer_center_oo /
#           obj_spatial_relation_oo / spatial_imagination_oc / spatial_imagination_oo
#
# 前置步骤:
# 1) 构建 GRPO 子数据集（已过滤验证集 id，避免泄露）:
#    python RL/build_grpo_dataset.py \
#      --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.jsonl \
#      --output-jsonl /data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl
#
# 2) 在单独终端启动 vLLM rollout server（只需启动一次，若报 Error: existing 表示已在运行，可忽略）:
#    CUDA_VISIBLE_DEVICES=6,7 \
#    swift rollout \
#      --model /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
#      --vllm_data_parallel_size 4
#
# 确认 rollout server 在端口 8000 正常运行后，再在本脚本所在终端执行 GRPO 训练。

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export WANDB_API_KEY='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'

# 模型与数据路径
BASE_MODEL="/data/dsq/Models/Qwen2.5-VL-3B-Instruct"
# 冷启动 LoRA checkpoint（SFT 输出）
SFT_ADAPTER="/home/lxp/Ground_reasoning/Models/rationale_sft/2e-4/"
GRPO_DATASET="/data/dsq/ScanNet/qa_jsonl/train.question.rl.grpo.jsonl"
GRPO_VAL_DATASET="/data/dsq/ScanNet/qa_jsonl/val.question.rl.grpo.jsonl"
OUTPUT_DIR="/home/lxp/Ground_reasoning/Models/rationale_grpo/2e-4"

# GRPO 训练超参（可按需调整）
NUM_EPOCHS=1
PER_DEVICE_BATCH=8  # 增大batch size以充分利用24GB显存
LR=1e-6
BETA=0.001
NUM_GENERATIONS=8   # 增加rollout数量，提升RL信号质量
# 增大单次生成的最大长度，减少 [PRED_RAW] 被截断的情况
MAX_COMPLETION_LEN=4096
LOG_STEPS=10
GRADIENT_ACCUMULATION=2
DATALOADER_WORKERS=8     # 增加数据加载worker数量，加速数据加载

swift rlhf \
  --rlhf_type grpo \
  --model "${BASE_MODEL}" \
  --adapters "${SFT_ADAPTER}" \
  --torch_dtype bfloat16 \
  --tuner_type lora \
  --dataset "${GRPO_DATASET}" \
  --val_dataset "${GRPO_VAL_DATASET}" \
  --external_plugins RL/grpo_plugin_scanet.py \
  --reward_funcs external_scanet_acc format \
  --use_vllm false \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH} \
  --per_device_eval_batch_size ${PER_DEVICE_BATCH} \
  --learning_rate ${LR} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
  --save_strategy 'steps' \
  --eval_strategy 'steps' \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 5 \
  --logging_steps ${LOG_STEPS} \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.01 \
  --dataloader_num_workers ${DATALOADER_WORKERS} \
  --num_generations ${NUM_GENERATIONS} \
  --max_completion_length ${MAX_COMPLETION_LEN} \
  --system "You are a 3D spatial understanding expert." \
  --log_completions true \
  --report_to none \
  --num_iterations 1 \
  --async_generate false \
  --beta ${BETA}

echo "GRPO training finished. Output dir: ${OUTPUT_DIR}"

