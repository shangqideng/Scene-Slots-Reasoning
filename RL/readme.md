- 删除没有rationale的行
python RL/clean_incomplete_jsonl.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.cleaned.jsonl
- 将清理后的文件重命名回原文件名
mv /data/dsq/ScanNet/qa_jsonl/all.question.rl.cleaned.jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl
- 使用新模型继续处理
python RL/generate_rationale.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --prompt-template RL/rationale_prompt_template.txt \
  --model qwen-plus-2025-12-01 \
  --num-workers 64


- 查看jsonl文件行数：wc -l