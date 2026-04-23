import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multi_llm.prompt import prompts
from multi_llm.utils import country_capitalized_mapping
from label_utils import extract_label


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_after_keyword(text: str, keyword: str) -> str:
    if keyword in text:
        return text.rsplit(keyword, 1)[-1].strip()
    return text.strip()


def parse_after_any_keyword(text: str, keywords: List[str]) -> str:
    parsed = text.strip()
    for keyword in keywords:
        if keyword in parsed:
            parsed = parsed.rsplit(keyword, 1)[-1].strip()
    return parsed


def maybe_apply_chat_template(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_id: str, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = dict(device_map="auto", trust_remote_code=True)
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    prompt_text = maybe_apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt_text):
        text = text[len(prompt_text):].strip()
    return text


def format_eta(start: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    elapsed = time.time() - start
    per_item = elapsed / done
    remaining = per_item * max(total - done, 0)
    finish = datetime.now() + timedelta(seconds=remaining)
    return finish.strftime("%Y-%m-%d %H:%M:%S")


def debate_one(
    record: dict,
    model_a: Tuple,
    model_b: Tuple,
    alias_a: str,
    alias_b: str,
    max_new_tokens: int,
) -> Dict[str, str]:
    tok_a, llm_a = model_a
    tok_b, llm_b = model_b

    country = country_capitalized_mapping[record["Country"]]
    story = record["Story"]
    rot = record["Rule-of-Thumb"]

    p1_a = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
    p1_b = p1_a

    a1_raw = generate(tok_a, llm_a, p1_a, max_new_tokens)
    b1_raw = generate(tok_b, llm_b, p1_b, max_new_tokens)
    a1 = parse_after_keyword(a1_raw, "Answer:")
    b1 = parse_after_keyword(b1_raw, "Answer:")

    p2_a = (
        prompts["prompt_2"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", a1)
        .replace("{{other_response}}", b1)
    )
    p2_b = (
        prompts["prompt_2"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", b1)
        .replace("{{other_response}}", a1)
    )

    a2_raw = generate(tok_a, llm_a, p2_a, max_new_tokens)
    b2_raw = generate(tok_b, llm_b, p2_b, max_new_tokens)
    a2 = parse_after_keyword(a2_raw, "Response:")
    b2 = parse_after_keyword(b2_raw, "Response:")

    p3_a = (
        prompts["prompt_3"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", a1)
        .replace("{{other_response}}", b1)
        .replace("{{your_feedback}}", a2)
        .replace("{{other_feedback}}", b2)
    )
    p3_b = (
        prompts["prompt_3"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", b1)
        .replace("{{other_response}}", a1)
        .replace("{{your_feedback}}", b2)
        .replace("{{other_feedback}}", a2)
    )

    a3_raw = generate(tok_a, llm_a, p3_a, max_new_tokens)
    b3_raw = generate(tok_b, llm_b, p3_b, max_new_tokens)
    a3 = parse_after_any_keyword(a3_raw, ["Answer (Yes, No or Neither):", "Answer:", "Answer", "Label:"])
    b3 = parse_after_any_keyword(b3_raw, ["Answer (Yes, No or Neither):", "Answer:", "Answer", "Label:"])

    return {
        f"{alias_a}_1": a1,
        f"{alias_b}_1": b1,
        f"{alias_a}_2": a2,
        f"{alias_b}_2": b2,
        f"{alias_a}_final_raw": a3_raw,
        f"{alias_b}_final_raw": b3_raw,
        f"{alias_a}_final": extract_label(a3),
        f"{alias_b}_final": extract_label(b3),
    }


def load_done_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full open-model multi-LLM debate on a JSONL dataset.")
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_a", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model_b", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--alias_a", default="qwen3")
    parser.add_argument("--alias_b", default="qwen15")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load models in 4-bit quantization for faster inference (requires bitsandbytes)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(input_path))
    if args.max_examples > 0:
        records = records[: args.max_examples]

    total = len(records)
    if total == 0:
        raise ValueError("No records found.")

    done = load_done_count(output_path) if args.resume else 0

    print(f"Loading model A: {args.model_a}")
    model_a = load_model(args.model_a, load_in_4bit=args.load_in_4bit)
    print(f"Loading model B: {args.model_b}")
    model_b = load_model(args.model_b, load_in_4bit=args.load_in_4bit)

    start = time.time()
    for i in range(done, total):
        row = records[i]
        per_start = time.time()
        try:
            debate = debate_one(
                record=row,
                model_a=model_a,
                model_b=model_b,
                alias_a=args.alias_a,
                alias_b=args.alias_b,
                max_new_tokens=args.max_new_tokens,
            )
            error_text = ""
        except Exception as exc:
            # Keep long runs alive by recording failures and moving to the next row.
            debate = {
                f"{args.alias_a}_1": "",
                f"{args.alias_b}_1": "",
                f"{args.alias_a}_2": "",
                f"{args.alias_b}_2": "",
                f"{args.alias_a}_final": "invalid",
                f"{args.alias_b}_final": "invalid",
            }
            error_text = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = {
            "Country": row.get("Country", ""),
            "Story": row.get("Story", ""),
            "Rule-of-Thumb": row.get("Rule-of-Thumb", ""),
            "Gold Label": row.get("Gold Label", ""),
            "error": error_text,
        }
        out.update(debate)
        append_jsonl(output_path, out)

        gold = extract_label(str(row.get("Gold Label", "")))
        a_pred = out[f"{args.alias_a}_final"]
        b_pred = out[f"{args.alias_b}_final"]
        elapsed = time.time() - per_start
        eta = format_eta(start, i + 1, total)
        print(
            f"[{i + 1}/{total}] gold={gold} {args.alias_a}={a_pred} {args.alias_b}={b_pred} "
            f"item_time={elapsed:.2f}s ETA={eta}"
        )

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
