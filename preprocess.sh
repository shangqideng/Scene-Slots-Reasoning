torchrun --nproc_per_node=8 pointgraph/run_scene_graph.py \
  --jsonl /data/dsq/ScanNet/qa_jsonl/all.jsonl \
  --out-jsonl /data/dsq/ScanNet/qa_jsonl/all.qwen_scene_slot.jsonl \
  --object-mode qwen


# 感觉question解析的不一定好，后续可换成qwen解析question
# torchrun --nproc_per_node=8 pointgraph/run_scene_graph.py \
#   --jsonl /data/dsq/ScanNet/qa_jsonl/all.jsonl \
#   --out-jsonl /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot.jsonl \
#   --object-mode question
