import os
import json
import argparse
import base64
import time
from typing import Dict, Any, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_qa(sample: Dict[str, Any]) -> Tuple[str | None, str | None]:
    """
    与 8infer.py 中逻辑保持一致：从 conversations 里抽取首个 human / gpt。
    """
    conversations = sample.get("conversations", [])
    question = None
    answer = None
    for item in conversations:
        if item.get("from") == "human" and question is None:
            question = item.get("value")
        elif item.get("from") == "gpt" and answer is None:
            answer = item.get("value")
    return question, answer


def _resize_image(image: Image.Image, image_size: int) -> Image.Image:
    """
    将图像最长边缩放到 image_size，保持长宽比。
    """
    if image_size <= 0:
        return image
    w, h = image.size
    if max(w, h) <= image_size:
        return image
    scale = image_size / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.BICUBIC)


def encode_image_as_data_url(image_path: str, image_size: int) -> str | None:
    if not os.path.isfile(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img = _resize_image(img, image_size)
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def call_vision_chat_completion(
    api_base: str,
    api_key: str | None,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    timeout: float = 60.0,
    retry: int = 3,
) -> str:
    """
    通用 OpenAI-compatible /v1/chat/completions 调用。
    对于 InternVL / Qwen2.5-VL 等，只要服务端兼容该协议即可直接使用。
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    last_err = None
    for _ in range(max(1, retry)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(1.0)
                continue
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                last_err = RuntimeError(f"No choices in response: {data}")
                time.sleep(1.0)
                continue
            content = choices[0].get("message", {}).get("content", "")
            return str(content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    return f"[API_ERROR] {last_err}"


def _process_single_sample(
    idx: int,
    sample: Dict[str, Any],
    model: str,
    api_base: str,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
    retry: int,
    image_size: int,
) -> Tuple[int, Dict[str, Any] | None]:
    """
    单个样本的推理逻辑，用于多线程调用。
    """
    question, answer = build_qa(sample)
    image_list = sample.get("image", [])
    if not question or not answer or not image_list:
        print(f"[API Infer] Skip sample {idx} due to missing question/answer/image")
        return idx, None

    image_path = image_list[0]
    data_url = encode_image_as_data_url(image_path, image_size=image_size)
    if data_url is None:
        print(f"[API Infer] Skip sample {idx} because image not found or invalid: {image_path}")
        return idx, None

    # OpenAI-compatible vision 格式：text + image_url
    user_content = [
        {
            "type": "text",
            "text": (
                f"{question}\n\n"
                "Please answer concisely and directly in English, "
                "following exactly the answer format required in the question. "
                "Do not add any explanation or extra words."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        },
    ]

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    resp_text = call_vision_chat_completion(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
        retry=retry,
    )

    record = {
        "id": sample.get("id"),
        "question": question,
        "answer": answer,
        "response": resp_text,
        "type": sample.get("type"),
        "image": image_path,
    }
    return idx, record


def infer_one_file_api(
    in_jsonl: str,
    out_jsonl: str,
    model: str,
    api_base: str,
    api_key: str | None,
    max_tokens: int,
    timeout: float,
    retry: int,
    image_size: int,
    num_workers: int,
) -> None:
    print(f"[API Infer] Loading data from {in_jsonl}")
    data = load_jsonl(in_jsonl)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    total = len(data)
    written = 0

    num_workers = max(1, int(num_workers))
    print(f"[API Infer] Using {num_workers} workers, image_size={image_size}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor, open(
        out_jsonl, "w", encoding="utf-8"
    ) as f_out:
        futures = []
        for idx, sample in enumerate(data):
            futures.append(
                executor.submit(
                    _process_single_sample,
                    idx,
                    sample,
                    model,
                    api_base,
                    api_key,
                    max_tokens,
                    timeout,
                    retry,
                    image_size,
                )
            )

        for n, fut in enumerate(as_completed(futures), start=1):
            idx, record = fut.result()
            if record is None:
                continue
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if n % 50 == 0 or n == len(futures):
                print(f"[API Infer] processed={n}/{len(futures)}, written={written}")

    print(f"[API Infer] Saved predictions to {out_jsonl}, total written={written}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 OpenAI-compatible API（InternVL2.5 / Qwen2.5-VL 等）进行多模态 QA 推理，"
        "输出与 9score_only_vllm.py 兼容的 *.pred.jsonl 文件。"
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        required=True,
        help="评测集 jsonl，格式与 8infer.py 相同（包含 conversations 与 image 字段）。",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="输出目录，例如 /data/dsq/ScanNet/qa_jsonl/infer/api/Qwen2.5-VL-7B",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="API 模型名称，例如 InternVL2.5-2B 或 Qwen2.5-VL-7B",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="val",
        help="输出文件前缀名，最终文件名为 <split-name>.pred.jsonl，默认为 val。",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL，例如 https://api.openai.com 或本地部署地址；"
        "默认为环境变量 OPENAI_API_BASE 或 http://localhost:8000。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key；默认为环境变量 OPENAI_API_KEY。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="生成最大新 token 数。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="单次请求超时时间（秒）。",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=3,
        help="请求失败重试次数。",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="统一缩放后的图像最长边尺寸。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="并发请求线程数。",
    )
    args = parser.parse_args()

    api_base = args.api_base or os.environ.get("OPENAI_API_BASE") or "http://localhost:8000"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    pred_jsonl = os.path.join(args.out_dir, f"{args.split_name}.pred.jsonl")

    infer_one_file_api(
        in_jsonl=args.val_jsonl,
        out_jsonl=pred_jsonl,
        model=args.model,
        api_base=api_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retry=args.retry,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

