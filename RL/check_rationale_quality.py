"""
检查生成的rationale jsonl文件的质量。

使用方法：
# 基础检查（不使用LLM）
python RL/check_rationale_quality.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --output-json /home/lxp/Ground_reasoning/RL/quality_check_result.json \
  --num-workers 8

# 使用LLM进行深度检查
python RL/check_rationale_quality.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rl.jsonl \
  --output-json /home/lxp/Ground_reasoning/RL/quality_check_result.json \
  --llm-path /data3/lxp/Models/Qwen2.5-14B-Instruct-1M/ \
  --use-llm \
  --num-workers 8 \
  --sample-check-ratio 0.1  # 只检查10%的样本（LLM检查较慢）
  
检查结果会保存到指定的JSON文件，包含：
总体质量分数：0-100
统计信息（错误、警告、问题分布）
每个样本的详细检查结果
LLM检查结果（如果启用）
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Any, Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def get_qa(sample):
    """从样本中提取问题和答案"""
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    if question is None or answer is None:
        return None, None
    return str(question).strip(), str(answer).strip()


def get_scene_slot(sample):
    """从样本中提取scene_slot"""
    return sample.get("scene_slot")


def check_format(sample, line_num):
    """检查格式正确性"""
    errors = []
    warnings = []
    
    # 检查必需字段
    required_fields = ["id", "conversations", "rationale"]
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
    
    # 检查rationale字段
    if "rationale" in sample:
        rationale = sample["rationale"]
        if not rationale or not isinstance(rationale, str):
            errors.append("rationale field is empty or not a string")
        elif len(rationale.strip()) < 10:
            warnings.append("rationale is too short (less than 10 characters)")
    
    # 检查conversations格式
    if "conversations" in sample:
        conversations = sample["conversations"]
        if not isinstance(conversations, list):
            errors.append("conversations is not a list")
        else:
            has_human = False
            has_gpt = False
            for item in conversations:
                if not isinstance(item, dict):
                    errors.append("conversation item is not a dict")
                    continue
                if item.get("from") == "human":
                    has_human = True
                elif item.get("from") == "gpt":
                    has_gpt = True
            if not has_human:
                errors.append("No human message in conversations")
            if not has_gpt:
                errors.append("No gpt message in conversations")
    
    # 检查scene_slot
    if "scene_slot" in sample:
        scene_slot = sample["scene_slot"]
        if scene_slot is not None and not isinstance(scene_slot, list):
            warnings.append("scene_slot is not a list")
    
    return errors, warnings


def check_consistency_basic(sample):
    """基础一致性检查（不需要LLM）"""
    issues = []
    
    question, answer = get_qa(sample)
    rationale = sample.get("rationale", "")
    scene_slot = get_scene_slot(sample)
    
    if not question or not answer or not rationale:
        return issues
    
    # 检查rationale是否提到question中的关键词
    question_lower = question.lower()
    rationale_lower = rationale.lower()
    
    # 提取问题中的关键对象
    if "of " in question_lower:
        obj_keywords = []
        parts = question_lower.split("of ")
        if len(parts) > 1:
            obj_part = parts[1].split()[0] if parts[1].split() else ""
            if obj_part:
                obj_keywords.append(obj_part)
                # 也检查复数形式
                if obj_part.endswith("s"):
                    obj_keywords.append(obj_part[:-1])
                else:
                    obj_keywords.append(obj_part + "s")
        
        # 检查rationale是否提到对象
        found = False
        for keyword in obj_keywords:
            if keyword in rationale_lower:
                found = True
                break
        if not found and obj_keywords:
            issues.append(f"Rationale may not mention the object from question")
    
    # 检查rationale是否提到scene_slot
    if scene_slot and isinstance(scene_slot, list) and len(scene_slot) > 0:
        slot_mentioned = False
        slot_text = json.dumps(scene_slot, ensure_ascii=False).lower()
        # 检查是否提到bbox_3d, center_3d等关键字段
        key_fields = ["bbox_3d", "center_3d", "2dmask_bbox"]
        for field in key_fields:
            if field in slot_text and field.replace("_", " ") in rationale_lower:
                slot_mentioned = True
                break
        if not slot_mentioned:
            issues.append("Rationale may not reference scene_slot information")
    
    # 检查rationale长度合理性
    if len(rationale) < 100:
        issues.append("Rationale is very short, may lack detail")
    elif len(rationale) > 5000:
        issues.append("Rationale is very long, may contain unnecessary content")
    
    return issues


def check_with_llm(sample, tokenizer, model, device):
    """使用LLM检查rationale与answer的一致性"""
    question, answer = get_qa(sample)
    rationale = sample.get("rationale", "")
    scene_slot = get_scene_slot(sample)
    
    if not question or not answer or not rationale:
        return {"consistent": False, "reason": "Missing required fields"}
    
    # 构建检查prompt
    slot_text = json.dumps(scene_slot, ensure_ascii=False) if scene_slot else "None"
    prompt = f"""You are a quality checker. Check if the rationale correctly supports the given answer.

Question: {question}

Scene Slot: {slot_text}

Answer: {answer}

Rationale: {rationale}

Please check:
1. Does the rationale correctly explain how to derive the answer from the scene_slot?
2. Is the rationale logically consistent with the answer?
3. Are there any contradictions between the rationale and the answer?

Respond in JSON format:
{{
    "consistent": true/false,
    "reason": "brief explanation",
    "issues": ["issue1", "issue2"]
}}"""
    
    try:
        # 使用LLM生成检查结果
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 构建输入
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                return {"consistent": True, "reason": "Could not parse LLM response", "raw_response": response}
        except json.JSONDecodeError:
            return {"consistent": True, "reason": "LLM response not in JSON format", "raw_response": response}
            
    except Exception as e:
        return {"consistent": False, "reason": f"LLM check failed: {str(e)}"}


def process_sample(args_tuple):
    """处理单个样本的检查"""
    line_num, sample, use_llm, tokenizer, model, device = args_tuple
    
    result = {
        "line_num": line_num,
        "id": sample.get("id", "unknown"),
        "format_errors": [],
        "format_warnings": [],
        "consistency_issues": [],
        "llm_check": None,
    }
    
    # 格式检查
    format_errors, format_warnings = check_format(sample, line_num)
    result["format_errors"] = format_errors
    result["format_warnings"] = format_warnings
    
    # 基础一致性检查
    consistency_issues = check_consistency_basic(sample)
    result["consistency_issues"] = consistency_issues
    
    # LLM检查（如果启用）
    if use_llm and model is not None:
        try:
            llm_result = check_with_llm(sample, tokenizer, model, device)
            result["llm_check"] = llm_result
        except Exception as e:
            result["llm_check"] = {"consistent": False, "reason": f"LLM check error: {str(e)}"}
    
    return result


def check_jsonl_quality(
    input_path,
    output_json,
    llm_path=None,
    num_workers=8,
    sample_check_ratio=1.0,
    use_llm=False,
):
    """检查jsonl文件质量"""
    
    # 加载LLM（如果提供）
    tokenizer = None
    model = None
    device = None
    
    if use_llm and llm_path:
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, skipping LLM checks")
            use_llm = False
        elif not os.path.exists(llm_path):
            print(f"Warning: LLM path not found: {llm_path}, skipping LLM checks")
            use_llm = False
        else:
            print(f"Loading LLM from {llm_path}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                device = next(model.parameters()).device
                print(f"LLM loaded on device: {device}")
            except Exception as e:
                print(f"Warning: Failed to load LLM: {e}, skipping LLM checks")
                use_llm = False
                tokenizer = None
                model = None
    
    # 读取所有样本
    print("Reading jsonl file...")
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                samples.append((line_num, sample))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
    
    total_samples = len(samples)
    print(f"Found {total_samples} samples")
    
    # 决定检查哪些样本
    if sample_check_ratio < 1.0:
        import random
        random.seed(42)
        num_to_check = int(total_samples * sample_check_ratio)
        samples = random.sample(samples, num_to_check)
        print(f"Sampling {len(samples)} samples for checking (ratio: {sample_check_ratio})")
    
    # 统计信息
    stats = {
        "total_samples": total_samples,
        "checked_samples": len(samples),
        "format_errors": defaultdict(int),
        "format_warnings": defaultdict(int),
        "consistency_issues": defaultdict(int),
        "llm_check_results": {"consistent": 0, "inconsistent": 0, "failed": 0},
    }
    
    # 处理样本
    results = []
    processed_count = 0
    
    if num_workers > 1 and len(samples) > 1:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_sample,
                    (line_num, sample, use_llm, tokenizer, model, device)
                ): line_num
                for line_num, sample in samples
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    processed_count += 1
                    
                    # 更新统计
                    for error in result["format_errors"]:
                        stats["format_errors"][error] += 1
                    for warning in result["format_warnings"]:
                        stats["format_warnings"][warning] += 1
                    for issue in result["consistency_issues"]:
                        stats["consistency_issues"][issue] += 1
                    
                    if result["llm_check"]:
                        if result["llm_check"].get("consistent", True):
                            stats["llm_check_results"]["consistent"] += 1
                        else:
                            stats["llm_check_results"]["inconsistent"] += 1
                    elif use_llm:
                        stats["llm_check_results"]["failed"] += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Checked {processed_count}/{len(samples)} samples...")
                except Exception as e:
                    print(f"Error processing sample: {e}")
    else:
        # 单线程处理
        for line_num, sample in samples:
            result = process_sample((line_num, sample, use_llm, tokenizer, model, device))
            results.append(result)
            processed_count += 1
            
            # 更新统计
            for error in result["format_errors"]:
                stats["format_errors"][error] += 1
            for warning in result["format_warnings"]:
                stats["format_warnings"][warning] += 1
            for issue in result["consistency_issues"]:
                stats["consistency_issues"][issue] += 1
            
            if result["llm_check"]:
                if result["llm_check"].get("consistent", True):
                    stats["llm_check_results"]["consistent"] += 1
                else:
                    stats["llm_check_results"]["inconsistent"] += 1
            elif use_llm:
                stats["llm_check_results"]["failed"] += 1
            
            if processed_count % 50 == 0:
                print(f"Checked {processed_count}/{len(samples)} samples...")
    
    # 计算总体质量分数
    total_errors = sum(len(r["format_errors"]) for r in results)
    total_warnings = sum(len(r["format_warnings"]) for r in results)
    total_issues = sum(len(r["consistency_issues"]) for r in results)
    
    quality_score = 100.0
    if len(results) > 0:
        quality_score -= (total_errors / len(results)) * 50  # 每个错误扣50分
        quality_score -= (total_warnings / len(results)) * 10  # 每个警告扣10分
        quality_score -= (total_issues / len(results)) * 5  # 每个问题扣5分
        quality_score = max(0, quality_score)
    
    # 汇总结果
    summary = {
        "check_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_path,
        "total_samples": total_samples,
        "checked_samples": len(samples),
        "quality_score": round(quality_score, 2),
        "statistics": {
            "total_format_errors": total_errors,
            "total_format_warnings": total_warnings,
            "total_consistency_issues": total_issues,
            "error_distribution": dict(stats["format_errors"]),
            "warning_distribution": dict(stats["format_warnings"]),
            "issue_distribution": dict(stats["consistency_issues"]),
            "llm_check_results": stats["llm_check_results"],
        },
        "detailed_results": results,
    }
    
    # 保存结果
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Quality Check Summary")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Checked samples: {len(samples)}")
    print(f"Quality Score: {quality_score:.2f}/100")
    print(f"\nFormat Errors: {total_errors}")
    print(f"Format Warnings: {total_warnings}")
    print(f"Consistency Issues: {total_issues}")
    if use_llm:
        print(f"\nLLM Check Results:")
        print(f"  Consistent: {stats['llm_check_results']['consistent']}")
        print(f"  Inconsistent: {stats['llm_check_results']['inconsistent']}")
        print(f"  Failed: {stats['llm_check_results']['failed']}")
    print(f"\nResults saved to: {output_json}")
    
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Check quality of rationale jsonl file")
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Input jsonl file path to check"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON file path for check results"
    )
    parser.add_argument(
        "--llm-path",
        type=str,
        default=None,
        help="Path to local LLM for advanced checking (optional)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM-based consistency checking"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent workers for checking"
    )
    parser.add_argument(
        "--sample-check-ratio",
        type=float,
        default=1.0,
        help="Ratio of samples to check (0.0-1.0, default: 1.0 for all)"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    
    if not os.path.isfile(args.input_jsonl):
        raise RuntimeError(f"Input file not found: {args.input_jsonl}")
    
    check_jsonl_quality(
        input_path=args.input_jsonl,
        output_json=args.output_json,
        llm_path=args.llm_path,
        num_workers=args.num_workers,
        sample_check_ratio=args.sample_check_ratio,
        use_llm=args.use_llm,
    )


if __name__ == "__main__":
    main()
