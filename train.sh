#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

torchrun --nproc_per_node=6 7_1finetune_qwen2_5_vl_slot_noimg.py \
 --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.question_scene_slot_correct_rename_add2dmask.jsonl \
 --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
 --output-dir ./Models/question_scene_slot/3b_baseline_ft_lr2e-5_noimg \
 --lora \
 --lr 2e-5 \
 --epochs 2 \
 --per-device-batch 2 \
 --grad-accum 11 \
 --max-text-tokens 1024

torchrun --nproc_per_node=6 7_2finetune_qwen2_5_vl_slot_llm.py \
 --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.question_scene_slot_correct_rename_add2dmask.jsonl \
 --model-path /data3/lxp/Models/Qwen2.5-3B-Instruct \
 --output-dir ./Models/question_scene_slot/3b_baseline_ft_lr2e-5_llm \
 --lora \
 --lr 2e-5 \
 --epochs 2 \
 --per-device-batch 2 \
 --grad-accum 11 \
 --max-text-tokens 1024


torchrun --nproc_per_node=6 7_3finetune_qwen2_5_vl_slot_randommask.py \
 --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.question_scene_slot_correct_rename_add2dmask.jsonl \
 --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
 --output-dir ./Models/question_scene_slot/3b_baseline_ft_lr2e-5_randommask \
 --lora \
 --lr 2e-5 \
 --epochs 2 \
 --per-device-batch 2 \
 --grad-accum 11 \
 --image-size 256 \
 --random-mask-ratio 0.25 \
 --max-text-tokens 1024

###
torchrun --nproc_per_node=6 7_4finetune_qwen2_5_vl_slot_qformer.py \
 --train-jsonl /data/dsq/ScanNet/qa_jsonl/train.question_scene_slot_correct_rename_add2dmask.jsonl \
 --model-path /data/dsq/Models/Qwen2.5-VL-3B-Instruct \
 --output-dir ./Models/question_scene_slot/3b_baseline_ft_lr2e-5_qformer \
 --lora \
 --lr 2e-5 \
 --epochs 2 \
 --per-device-batch 2 \
 --grad-accum 11 \
 --image-size 256 \
 --slot-qformer-source scene_slot \
 --max-text-tokens 1024
