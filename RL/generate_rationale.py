"""
从 jsonl 文件中读取数据，调用 API 生成 rationale，并保存到新的 jsonl 文件。

使用方法：
python ./RL/generate_rationale.py \
  --input-jsonl /data/dsq/ScanNet/qa_jsonl/all.question_scene_slot_correct_rename_add2dmask.jsonl \
  --output-jsonl /data/dsq/ScanNet/qa_jsonl/all.question.rationale.modified_ds.jsonl \
  --prompt-template RL/rationale_prompt_template.txt \
  --model deepseek-v3.2 \
  --num-workers 32 \ 
  --max-samples 50

deepseek-v3.2  qwen3.5-plus-2026-02-15
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI


def get_qa(sample):
    """从样本中提取问题和答案，与 7_3finetune_qwen2_5_vl_slot_randommask.py 保持一致"""
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
    """从样本中提取 scene_slot，与 7_3finetune_qwen2_5_vl_slot_randommask.py 保持一致"""
    return sample.get("scene_slot")


def load_prompt_template(template_path):
    """加载 prompt 模板"""
    if not os.path.isfile(template_path):
        raise RuntimeError(f"Prompt template not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(template, question, scene_slot, answer):
    """根据模板构建 prompt"""
    slot_text = json.dumps(scene_slot, ensure_ascii=False, indent=2)
    prompt = template.format(
        question=question,
        scene_slot=slot_text,
        answer=answer
    )
    return prompt


def call_api(client, model, prompt, max_retries=5, retry_delay=1.0):
    """Call API to generate rationale - each call is completely independent with no context interference.
    
    This function ensures that each API call is stateless and independent:
    - Creates a fresh messages list for each call
    - No context is shared between calls
    - Each call is a standalone request with no session or state retention
    - Uses exponential backoff for 429 rate limit errors
    """
    for attempt in range(max_retries):
        try:
            # Create a completely fresh messages list for each attempt
            # This ensures no context interference between calls
            messages = [
                {
                    "role": "user",
                    "content": str(prompt)  # Ensure prompt is a fresh string
                }
            ]
            
            # Each call is independent - no session, no context retention
            # OpenAI API is stateless by default, each request is isolated
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # Fresh messages list created for each call
                temperature=0.7,
            )
            rationale = completion.choices[0].message.content.strip()
            return rationale
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower() or "limit_requests" in error_str
            
            if attempt < max_retries - 1:
                if is_rate_limit:
                    # For 429 errors, use exponential backoff with longer wait times
                    # Wait time: 5s, 10s, 20s, 40s, 80s
                    wait_time = 5 * (2 ** attempt)
                    print(f"Rate limit error (attempt {attempt + 1}/{max_retries}), waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    # For other errors, use shorter exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
            else:
                print(f"API call finally failed after {max_retries} attempts: {e}")
                raise


def count_processed_lines(output_path):
    """Count how many lines have been processed in output file (including empty lines)"""
    if not os.path.isfile(output_path):
        return 0
    
    count = 0
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                count += 1  # Count all lines including empty ones
    except Exception as e:
        print(f"Warning: Failed to read output file for resume check: {e}")
        return 0
    
    return count


def get_processed_samples(output_path):
    """Get set of sample IDs that have been processed (have rationale field).
    
    Returns:
        set of sample IDs (as strings) that have rationale field
    """
    if not os.path.isfile(output_path):
        return set()
    
    processed_ids = set()
    
    try:
        with open(output_path, "r", encoding="utf-8") as f_out:
            for line in f_out:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "rationale" in obj and obj["rationale"]:
                        # Only use id for matching
                        sample_id = obj.get("id")
                        if sample_id:
                            processed_ids.add(str(sample_id))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Failed to read output file for resume check: {e}")
        return set()
    
    return processed_ids


def process_single_sample(args_tuple):
    """Process a single sample - completely independent, no context interference.
    
    Args:
        args_tuple: (line_num, obj, template, model, api_key, base_url)
    
    Returns:
        (line_num, obj, rationale, error) or None if skipped
    """
    line_num, obj, template, model, api_key, base_url = args_tuple
    
    # Create a completely independent client for this call
    # This ensures no context interference between concurrent calls
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Extract question and answer
    question, answer = get_qa(obj)
    if question is None or answer is None:
        return (line_num, obj, None, "missing_qa")
    
    # Extract scene_slot
    scene_slot = get_scene_slot(obj)
    if scene_slot is None:
        return (line_num, obj, None, "missing_slot")
    
    # Build prompt - fresh for each call
    prompt = build_prompt(template, question, scene_slot, answer)
    
    # Call API - completely independent, no context from other calls
    try:
        rationale = call_api(client, model, prompt)
        return (line_num, obj, rationale, None)
    except Exception as e:
        return (line_num, obj, None, str(e))


def process_jsonl(input_path, output_path, template_path, model, max_samples=None, skip_missing_slot=True, num_workers=32):
    """Process jsonl file, generate rationale and save to new file with resume support.
    
    Uses multi-threading for concurrent API calls to maximize speed.
    Each API call is completely independent with no context interference.
    
    Args:
        num_workers: Number of concurrent threads (default: 32, can be increased for 8 GPUs)
    """
    # Load template
    template = load_prompt_template(template_path)
    
    # Initialize API credentials
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set environment variable DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Check for resume: find which samples have been processed (only by id)
    processed_ids = get_processed_samples(output_path)
    if processed_ids:
        print(f"Resuming: Found {len(processed_ids)} already processed samples in output file")
    
    # Collect all tasks first
    tasks = []
    total_lines = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        # Collect all tasks, skipping already processed ones
        for line_num, line in enumerate(f_in, start=1):
            total_lines += 1
            
            # Skip empty lines - don't process them
            if not line.strip():
                continue
            
            # Parse JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_num} (JSON parse error): {e}")
                continue
            
            # Check if already processed (only by id)
            sample_id = obj.get("id")
            if sample_id and str(sample_id) in processed_ids:
                # Skip already processed sample
                continue
            
            # Check if valid sample
            question, answer = get_qa(obj)
            if question is None or answer is None:
                # Skip invalid samples - don't add to tasks
                continue
            
            scene_slot = get_scene_slot(obj)
            if scene_slot is None:
                if skip_missing_slot:
                    # Skip samples without scene_slot
                    continue
                else:
                    scene_slot = {}
            
            # Add to processing tasks
            tasks.append((line_num, obj, None, "process"))
    
    print(f"Total tasks to process: {len(tasks)}")
    
    # Filter out tasks that need API calls
    api_tasks = [(line_num, obj, template, model, api_key, base_url) 
                  for line_num, obj, _, status in tasks if status == "process"]
    
    # Apply max_samples limit if specified
    if max_samples and len(api_tasks) > max_samples:
        # Limit both api_tasks and tasks list
        process_task_line_nums = {task[0] for task in api_tasks[:max_samples]}
        # Filter tasks to only include those that will be processed
        tasks = [t for t in tasks if t[3] != "process" or t[0] in process_task_line_nums]
        api_tasks = api_tasks[:max_samples]
    
    print(f"Processing {len(api_tasks)} samples with {num_workers} concurrent workers...")
    
    # Initialize counters
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Load existing processed samples from output file (if exists) - only by id
    existing_results = {}  # id -> obj with rationale
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        if "rationale" in obj and obj["rationale"]:
                            sample_id = obj.get("id")
                            if sample_id:
                                existing_results[str(sample_id)] = obj
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to read existing output file: {e}")
    
    # Create output file - we'll rebuild it in input order
    # Read all input lines first
    all_input_data = []  # List of (line_num, line, parsed_obj or None)
    with open(input_path, "r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, start=1):
            if not line.strip():
                all_input_data.append((line_num, line, None))
            else:
                try:
                    obj = json.loads(line)
                    all_input_data.append((line_num, line, obj))
                except json.JSONDecodeError:
                    all_input_data.append((line_num, line, None))
    
    # Create initial output file with existing results only (samples with rationale)
    # We'll append new results as they are processed
    file_mode = "w"  # Start fresh, but we'll maintain order by rewriting
    with open(output_path, file_mode, encoding="utf-8") as f_out:
        for line_num, line, obj in all_input_data:
            if obj is None:
                # Skip empty lines and JSON errors - don't write them
                continue
            else:
                # Only write if we have existing result with rationale
                sample_id = obj.get("id")
                if sample_id and str(sample_id) in existing_results:
                    existing_obj = existing_results[str(sample_id)]
                    if "rationale" in existing_obj and existing_obj["rationale"]:
                        # Use existing result with rationale
                        f_out.write(json.dumps(existing_obj, ensure_ascii=False) + "\n")
                        f_out.flush()
                # Otherwise skip - will be processed and written later if successful
    
    # Create a mapping from line_num to position in all_input_data for ordered writing
    line_num_to_input_idx = {}
    for idx, (line_num, _, obj) in enumerate(all_input_data):
        if obj is not None:
            line_num_to_input_idx[line_num] = idx
    
    # Track which samples have been written (by line_num)
    written_samples = set()
    for line_num, _, obj in all_input_data:
        if obj is not None:
            sample_id = obj.get("id")
            if sample_id and str(sample_id) in existing_results:
                existing_obj = existing_results[str(sample_id)]
                if "rationale" in existing_obj and existing_obj["rationale"]:
                    written_samples.add(line_num)
    
    # Process with thread pool - each call is independent
    results = {}  # line_num -> (obj, rationale, error)
    
    # Create a lock for thread-safe file operations
    write_lock = Lock()
    
    def write_result_immediately(line_num, obj_result, rationale):
        """Write a successfully processed sample immediately, maintaining input order."""
        nonlocal written_samples
        
        with write_lock:
            # Add rationale to object
            obj_result["rationale"] = rationale
            
            # Read current file content
            current_lines = []
            if os.path.isfile(output_path):
                with open(output_path, "r", encoding="utf-8") as f_in:
                    current_lines = [line for line in f_in if line.strip()]
            
            # Create a mapping of existing samples by id
            existing_by_id = {}
            for line in current_lines:
                try:
                    existing_obj = json.loads(line)
                    existing_id = existing_obj.get("id")
                    if existing_id:
                        existing_by_id[str(existing_id)] = existing_obj
                except:
                    continue
            
            # Add new result
            sample_id = obj_result.get("id")
            if sample_id:
                existing_by_id[str(sample_id)] = obj_result
            
            # Rewrite file in input order
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line_num_check, _, obj_check in all_input_data:
                    if obj_check is None:
                        continue
                    sample_id_check = obj_check.get("id")
                    if sample_id_check and str(sample_id_check) in existing_by_id:
                        existing_obj = existing_by_id[str(sample_id_check)]
                        if "rationale" in existing_obj and existing_obj["rationale"]:
                            f_out.write(json.dumps(existing_obj, ensure_ascii=False) + "\n")
                f_out.flush()
            
            written_samples.add(line_num)
    
    # Use ThreadPoolExecutor for concurrent API calls
    # Each thread creates its own client, ensuring no context interference
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_line = {
            executor.submit(process_single_sample, task): task[0] 
            for task in api_tasks
        }
        
        # Collect results as they complete and write immediately
        for future in as_completed(future_to_line):
            line_num = future_to_line[future]
            try:
                result = future.result()
                if result:
                    line_num_result, obj, rationale, error = result
                    # Store result
                    with write_lock:
                        results[line_num_result] = (obj, rationale, error)
                    
                    if rationale:
                        # Successfully processed - write immediately
                        write_result_immediately(line_num_result, obj, rationale)
                        processed_count += 1
                        if processed_count % 50 == 0:
                            print(f"Processed {processed_count}/{len(api_tasks)} samples...")
                    elif error == "missing_qa" or error == "missing_slot":
                        skipped_count += 1
                    else:
                        error_count += 1
                        print(f"Error processing line {line_num_result}: {error}")
            except Exception as e:
                print(f"Unexpected error for line {line_num}: {e}")
                error_count += 1
    
    print(f"\nProcessing completed:")
    print(f"  Total lines processed: {total_lines}")
    print(f"  Successfully processed: {processed_count} samples")
    print(f"  Skipped: {skipped_count} samples")
    print(f"  Errors: {error_count} samples")
    print(f"  Output file: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate rationale from jsonl file")
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Input jsonl file path"
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output jsonl file path"
    )
    parser.add_argument(
        "--prompt-template",
        default="RL/rationale_prompt_template.txt",
        help="Prompt template file path"
    )
    parser.add_argument(
        "--model",
        default="qwen3-max-2026-01-23",
        help="API model name"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Number of concurrent threads for API calls (default: 32, increase for more speed)"
    )
    parser.set_defaults(skip_missing_slot=True)
    parser.add_argument(
        "--no-skip-missing-slot",
        action="store_false",
        dest="skip_missing_slot",
        help="Do not skip samples without scene_slot (default: skip)"
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    
    if not os.path.isfile(args.input_jsonl):
        raise RuntimeError(f"Input file not found: {args.input_jsonl}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    process_jsonl(
        input_path=args.input_jsonl,
        output_path=args.output_jsonl,
        template_path=args.prompt_template,
        model=args.model,
        max_samples=args.max_samples,
        skip_missing_slot=args.skip_missing_slot,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
