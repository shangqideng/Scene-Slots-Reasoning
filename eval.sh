#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python 9score_only_vllm.py \
#   --in-dir /data/dsq/ScanNet/qa_jsonl/infer/against_noisy/no_noisy \
#  --score-max-model-len 8192


python 9score_only_vllm.py \
  --in-dir /data/dsq/ScanNet/qa_jsonl/infer/against_noisy/add_noisy \
 --score-max-model-len 8192