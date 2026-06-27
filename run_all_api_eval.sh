#!/usr/bin/env bash

set -e

# 评测集与输出根目录
VAL_JSONL="/data/dsq/ScanNet/qa_jsonl/val.question_scene_slot_correct_rename_add2dmask.jsonl"
OUT_ROOT="/data/dsq/ScanNet/qa_jsonl/infer/api"

# 使用 vLLM 的 HuggingFace 模型名（与 vllm serve 保持一致）
MODELS=(
  "OpenGVLab/InternVL2_5-2B"
  "OpenGVLab/InternVL2_5-4B"
  "OpenGVLab/InternVL2_5-8B"
  "OpenGVLab/InternVL2_5-26B"
  "OpenGVLab/InternVL2_5-38B"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-32B-Instruct"
  "Qwen/Qwen2.5-VL-72B-Instruct"
)

echo "[RUN] val jsonl: ${VAL_JSONL}"
echo "[RUN] out root:  ${OUT_ROOT}"

for MODEL in "${MODELS[@]}"; do
  # 目录名中避免出现斜杠
  SAFE_NAME="${MODEL//\//_}"
  OUT_DIR="${OUT_ROOT}/${SAFE_NAME}"

  echo "=============================="
  echo "[RUN] Model: ${MODEL}"
  echo "[RUN] Out dir: ${OUT_DIR}"

  mkdir -p "${OUT_DIR}"

  echo "[RUN] Start vLLM server for ${MODEL}"
  # 在 8000 端口启动 vLLM OpenAI 兼容服务
  vllm serve "${MODEL}" --host 0.0.0.0 --port 8000 > "${OUT_DIR}/vllm.log" 2>&1 &
  VLLM_PID=$!

  # 等待模型加载完成（可根据机器性能调整）
  sleep 25

  # 1) API 推理，生成 <split>.pred.jsonl（默认 split=val）
  python 8api_infer.py \
    --val-jsonl "${VAL_JSONL}" \
    --out-dir "${OUT_DIR}" \
    --model "${MODEL}" \
    --split-name "val" \
    --max-tokens 64 \
    --image-size 256 \
    --num-workers 16

  echo "[RUN] Stop vLLM server for ${MODEL}"
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true

  # 2) 使用已有的 9score_only_vllm.py 做自动评分
  python 9score_only_vllm.py \
    --in-dir "${OUT_DIR}"
done

echo "[RUN] 所有模型 API 推理 + 评分已完成。"

