import json


SRC_PATH = "/data/dsq/ScanNet/qa_jsonl/train.question.rl.swift.short.jsonl"
DST_PATH = "/data/dsq/ScanNet/qa_jsonl/train.question.rl.swift.short.rmtag.jsonl"


def clean_think(text: str) -> str:
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    if text.endswith("</think>"):
        text = text[: -len("</think>")]
    return text


def clean_answer(text: str) -> str:
    if text.startswith("<answer>"):
        text = text[len("<answer>") :]
    if text.endswith("</answer>"):
        text = text[: -len("</answer>")]
    text = text.strip()
    if text.startswith("Answer:"):
        text = text[len("Answer:") :].lstrip()
    return text


def process_line(obj: dict) -> dict:
    messages = obj.get("messages", [])
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]

    if len(assistant_indices) >= 1:
        first = messages[assistant_indices[0]]
        first["loss"] = True
        contents = first.get("content", [])
        for c in contents:
            if c.get("type") == "text":
                c["text"] = clean_think(c.get("text", ""))
                break

    if len(assistant_indices) >= 2:
        second = messages[assistant_indices[1]]
        second["loss"] = True
        contents = second.get("content", [])
        for c in contents:
            if c.get("type") == "text":
                c["text"] = clean_answer(c.get("text", ""))
                break

    return obj


def main() -> None:
    with open(SRC_PATH, "r", encoding="utf-8") as fin, open(
        DST_PATH, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj = process_line(obj)
            json.dump(obj, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    main()

