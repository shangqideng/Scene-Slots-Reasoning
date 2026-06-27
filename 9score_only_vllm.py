'''
python 9score_only_vllm.py \
  --in-dir /data/dsq/ScanNet/qa_jsonl/infer/3b_baseline_ft_lr1e-5
'''
import os
import json
import gc
import argparse
from typing import List, Tuple, Dict
import re
from ast import literal_eval

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SCORE_LLM_PATH = "/data/dsq/Models/Qwen2.5-14B-Instruct-1M"
DTYPE = torch.bfloat16


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARN] JSON decode error in {path}:{lineno}: {e}. "
                    f"Line content (truncated): {line[:200]!r}"
                )
                continue
    return data


def find_pred_jsonl(in_dir: str, pred_name: str | None):
    if pred_name:
        path = os.path.join(in_dir, pred_name)
        if not os.path.isfile(path):
            raise RuntimeError(f"pred jsonl not found: {path}")
        return path
    candidates = [
        os.path.join(in_dir, name)
        for name in os.listdir(in_dir)
        if name.endswith(".pred.jsonl")
    ]
    if not candidates:
        raise RuntimeError(f"No *.pred.jsonl found in {in_dir}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple *.pred.jsonl found in {in_dir}, use --pred-name to specify."
        )
    return candidates[0]


def build_score_prompt(
    question: str, response: str, answer: str, tokenizer: AutoTokenizer
) -> str:
    messages = [
        {
            "role": "system",
            "content":
                "You are an intelligent grader for question-answer pairs. "
                "Follow the scoring rules strictly and return a Python dict only.",
        },
        {
            "role": "user",
            "content":
                "Please evaluate the following image-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {response}\n\n"
                "Scoring rules:\n"
                "1) If the answer is numeric (single number), compute relative error = |pred-gt|/max(|gt|,1e-6).\n"
                "   - If relative error > 0.4: pred='no', score=0.\n"
                "   - Else: pred='yes', score = 1 - (relative_error / 0.4). Score in [0,1].\n"
                "2) If the question is Yes/No or multiple-choice (A/B/C/D), score strictly:\n"
                "   - Correct: pred='yes', score=1. Incorrect: pred='no', score=0.\n"
                "3) For other cases, judge semantic correctness: correct -> score=1, incorrect -> score=0.\n\n"
                "Return ONLY a Python dictionary string: {'pred': 'yes'/'no', 'score': <float between 0 and 1>}.\n"
                "Do NOT output anything else."
        }
    ]
    '''
    "Scoring rules:\n"
    "1) If the answer is numeric (single number), compute relative error = |pred-gt|/max(|gt|,1e-6).\n"
    "   - If relative error > 0.4: pred='no', score=0.\n"
    "   - Else: pred='yes', score = 1 - (relative_error / 0.4). Score in [0,1].\n"
    "2) If the answer is numeric (three numbers for spatial_volume_infer, format: L,W,H in cm), score by per-dimension relative error.\n"
    "   - If pred cannot be parsed into exactly 3 numbers: pred='no', score=0.\n"
    "   - For each dimension i in {L,W,H}, compute re_i = |p_i-g_i|/max(|g_i|,1e-6).\n"
    "     * If re_i > 0.30: s_i=0. Else: s_i = 1 - (re_i / 0.30). Each s_i in [0,1].\n"
    "   - score = (s_L + s_W + s_H) / 3.\n"
    "   - If L <= W (violates length>width constraint): score *= 0.7.\n"
    "   - pred='yes' if score>0 else pred='no'.\n"
    "3) If the question is Yes/No or multiple-choice (A/B/C/D), score strictly:\n"
    "   - Correct: pred='yes', score=1. Incorrect: pred='no', score=0.\n"
    "4) For other cases, judge semantic correctness: correct -> score=1, incorrect -> score=0.\n\n"
    "Return ONLY a Python dictionary string: {'pred': 'yes'/'no', 'score': <float between 0 and 1>}.\n"
    '''
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def _get_conv_value(sample: dict, role: str) -> str | None:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        return None
    for item in conversations:
        if isinstance(item, dict) and item.get("from") == role:
            value = item.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def resolve_question(sample: dict) -> str:
    q = sample.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    q_conv = _get_conv_value(sample, "human")
    return q_conv if q_conv is not None else ""


def resolve_answer(sample: dict) -> str:
    a = sample.get("answer")
    if isinstance(a, str) and a.strip():
        return a.strip()
    a_conv = _get_conv_value(sample, "gpt")
    return a_conv if a_conv is not None else ""


_DICT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def parse_score_output_strict(text: str) -> Tuple[str, float, bool]:
    if not text:
        return "no", 0.0, False
    m = _DICT_RE.search(text.strip())
    if not m:
        return "no", 0.0, False
    blob = m.group(0)
    try:
        obj = literal_eval(blob)
    except Exception:
        return "no", 0.0, False
    if not isinstance(obj, dict):
        return "no", 0.0, False
    if "pred" not in obj or "score" not in obj:
        return "no", 0.0, False
    pred = str(obj["pred"]).strip().lower()
    if pred not in ("yes", "no"):
        return "no", 0.0, False
    score = obj["score"]
    if isinstance(score, str):
        try:
            score = float(score)
        except Exception:
            return "no", 0.0, False
    if isinstance(score, (int, float)):
        score_val = float(score)
    else:
        return "no", 0.0, False
    if score_val < 0.0 or score_val > 1.0:
        return "no", 0.0, False
    return pred, score_val, True


def parse_score_output_fallback(text: str) -> Tuple[str, float, bool]:
    if not text:
        return "no", 0.0, False
    t = text.strip().lower()
    pred = None
    if "yes" in t:
        pred = "yes"
    elif "no" in t:
        pred = "no"
    score = None
    for m in re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", t):
        try:
            score = float(m)
            break
        except Exception:
            continue
    if pred is None or score is None:
        return "no", 0.0, False
    return pred, score, True


def score_with_vllm_until_ok(
    prompts: List[str],
    vllm_engine: LLM,
    sampling_params: SamplingParams,
    max_retries: int = 2,
) -> List[Tuple[str, float, bool]]:
    n = len(prompts)
    results: List[Tuple[str, float, bool]] = [("no", 0.0, False)] * n
    pending: List[int] = list(range(n))
    tries = 0
    while pending and tries < max_retries:
        pending_prompts = [prompts[i] for i in pending]
        outputs = vllm_engine.generate(pending_prompts, sampling_params)
        new_pending: List[int] = []
        for out_idx, out in enumerate(outputs):
            global_i = pending[out_idx]
            gen_text = out.outputs[0].text if out.outputs else ""
            pred, score_val, ok = parse_score_output_strict(gen_text)
            if ok:
                results[global_i] = (pred, float(score_val), True)
            else:
                new_pending.append(global_i)
        pending = new_pending
        tries += 1
    if pending:
        pending_prompts = [prompts[i] for i in pending]
        outputs = vllm_engine.generate(pending_prompts, sampling_params)
        for out_idx, out in enumerate(outputs):
            global_i = pending[out_idx]
            gen_text = out.outputs[0].text if out.outputs else ""
            pred, score_val, ok = parse_score_output_fallback(gen_text)
            if ok:
                results[global_i] = (pred, float(score_val), True)
    return results


def score_only(
    in_dir: str,
    pred_name: str | None,
    tp: int,
    gpu_mem_util: float,
    max_model_len: int,
    batch_size: int,
    max_tokens: int,
    max_retries: int,
    debug_print_n: int,
):
    pred_jsonl = find_pred_jsonl(in_dir, pred_name)
    scored_jsonl = pred_jsonl.replace(".pred.jsonl", ".scored.jsonl")
    metrics_json = pred_jsonl.replace(".pred.jsonl", ".metrics.json")

    data = load_jsonl(pred_jsonl)
    print(f"[Score] Loaded {len(data)} samples from {pred_jsonl}")

    score_tokenizer = AutoTokenizer.from_pretrained(SCORE_LLM_PATH)
    score_tokenizer.padding_side = "left"
    llm = LLM(
        model=SCORE_LLM_PATH,
        dtype=DTYPE,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        stop=None,
    )

    all_scores: List[float] = []
    all_correct: List[int] = []
    type_stats: Dict[str, Dict[str, List[float]]] = {}

    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        prompts = [
            build_score_prompt(
                resolve_question(r),
                str(r.get("response", "")).strip(),
                resolve_answer(r),
                score_tokenizer,
            )
            for r in batch
        ]
        batch_results: List[Tuple[str, float, bool]] = score_with_vllm_until_ok(
            prompts, llm, sampling_params, max_retries=max_retries
        )
        outputs = llm.generate(prompts, sampling_params)
        for idx, (r, (pred_flag, score, ok), out) in enumerate(zip(batch, batch_results, outputs)):
            if debug_print_n > 0:
                print("=== Score Debug ===")
                print("question:", resolve_question(r))
                print("answer:", resolve_answer(r))
                print("response:", r.get("response"))
                print("raw_output:", out.outputs[0].text if out.outputs else "")
                print("parsed:", pred_flag, score, ok)
                print("=== End Debug ===")
                debug_print_n -= 1
                if debug_print_n == 0:
                    return
            if not ok:
                continue
            r["score_pred"] = pred_flag
            r["score"] = float(score)
            all_scores.append(float(score))
            all_correct.append(1 if pred_flag == "yes" else 0)
            t = r.get("type") or "unknown"
            if t not in type_stats:
                type_stats[t] = {"scores": [], "correct": [], "total": []}
            type_stats[t]["scores"].append(float(score))
            type_stats[t]["correct"].append(1 if pred_flag == "yes" else 0)
        for r in batch:
            t = r.get("type") or "unknown"
            if t not in type_stats:
                type_stats[t] = {"scores": [], "correct": [], "total": []}
            type_stats[t]["total"].append(1.0)
        print(f"[Score] {min(start+batch_size, len(data))}/{len(data)}")

    metrics = {
        "count": len(data),
        "scored_count": len(all_scores),
        "main_mean_score": float(sum(all_scores) / len(all_scores)) if all_scores else None,
        "main_acc_yes": float(sum(all_correct) / len(all_correct)) if all_correct else None,
    }
    type_metrics = {}
    for t, s in type_stats.items():
        total = len(s["total"])
        scored_count = len(s["scores"])
        mean_score = float(sum(s["scores"]) / scored_count) if scored_count else None
        acc_yes = float(sum(s["correct"]) / scored_count) if scored_count else None
        type_metrics[t] = {
            "count": total,
            "scored_count": scored_count,
            "main_mean_score": mean_score,
            "main_acc_yes": acc_yes,
        }
    metrics["by_type"] = type_metrics

    os.makedirs(os.path.dirname(scored_jsonl), exist_ok=True)
    with open(scored_jsonl, "w", encoding="utf-8") as f:
        for r in data:
            if "score_pred" not in r or "score" not in r:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Score] Saved scored jsonl to {scored_jsonl}")
    print(f"[Score] Saved metrics to {metrics_json}")

    del llm, score_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--pred-name", type=str, default=None)
    parser.add_argument("--score-tp", type=int, default=8)
    parser.add_argument("--score-gpu-mem-util", type=float, default=0.6)
    parser.add_argument("--score-max-model-len", type=int, default=4096)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--score-max-tokens", type=int, default=32)
    parser.add_argument("--score-max-retries", type=int, default=3)
    parser.add_argument("--score-debug-n", type=int, default=0)
    args = parser.parse_args()

    score_only(
        in_dir=args.in_dir,
        pred_name=args.pred_name,
        tp=args.score_tp,
        gpu_mem_util=args.score_gpu_mem_util,
        max_model_len=args.score_max_model_len,
        batch_size=args.score_batch_size,
        max_tokens=args.score_max_tokens,
        max_retries=args.score_max_retries,
        debug_print_n=args.score_debug_n,
    )


if __name__ == "__main__":
    main()
