from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_llm.prompt import prompts
from multi_llm.utils import country_capitalized_mapping
from scripts.label_utils import extract_label, strip_answer_prefix


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def load_hf_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def hf_generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def ollama_generate(model: str, prompt: str, host: str, max_new_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
        },
    }
    request = Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", "")).strip()


def format_eta(start: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    per_item = (time.time() - start) / done
    finish = datetime.now() + timedelta(seconds=per_item * max(total - done, 0))
    return finish.strftime("%Y-%m-%d %H:%M:%S")


def debate_one(
    record: dict,
    hf_model,
    hf_alias: str,
    ollama_model: str,
    ollama_alias: str,
    ollama_host: str,
    max_new_tokens: int,
    timeout: int,
) -> dict:
    tokenizer, model = hf_model
    country = country_capitalized_mapping[record["Country"]]
    story = record["Story"]
    rot = record["Rule-of-Thumb"]

    p1 = prompts["prompt_1"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
    hf_1_raw = hf_generate(tokenizer, model, p1, max_new_tokens)
    ol_1_raw = ollama_generate(ollama_model, p1, ollama_host, max_new_tokens, timeout)
    hf_1 = strip_answer_prefix(hf_1_raw, ("Answer:",))
    ol_1 = strip_answer_prefix(ol_1_raw, ("Answer:",))

    p2_hf = (
        prompts["prompt_2"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", hf_1)
        .replace("{{other_response}}", ol_1)
    )
    p2_ol = (
        prompts["prompt_2"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", ol_1)
        .replace("{{other_response}}", hf_1)
    )
    hf_2_raw = hf_generate(tokenizer, model, p2_hf, max_new_tokens)
    ol_2_raw = ollama_generate(ollama_model, p2_ol, ollama_host, max_new_tokens, timeout)
    hf_2 = strip_answer_prefix(hf_2_raw, ("Response:",))
    ol_2 = strip_answer_prefix(ol_2_raw, ("Response:",))

    p3_hf = (
        prompts["prompt_3"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", hf_1)
        .replace("{{other_response}}", ol_1)
        .replace("{{your_feedback}}", hf_2)
        .replace("{{other_feedback}}", ol_2)
    )
    p3_ol = (
        prompts["prompt_3"]
        .replace("{{country}}", country)
        .replace("{{story}}", story)
        .replace("{{rot}}", rot)
        .replace("{{your_response}}", ol_1)
        .replace("{{other_response}}", hf_1)
        .replace("{{your_feedback}}", ol_2)
        .replace("{{other_feedback}}", hf_2)
    )
    hf_final_raw = hf_generate(tokenizer, model, p3_hf, max_new_tokens)
    ol_final_raw = ollama_generate(ollama_model, p3_ol, ollama_host, max_new_tokens, timeout)

    return {
        f"{hf_alias}_1": hf_1,
        f"{ollama_alias}_1": ol_1,
        f"{hf_alias}_2": hf_2,
        f"{ollama_alias}_2": ol_2,
        f"{hf_alias}_final_raw": hf_final_raw,
        f"{ollama_alias}_final_raw": ol_final_raw,
        f"{hf_alias}_final": extract_label(hf_final_raw),
        f"{ollama_alias}_final": extract_label(ol_final_raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HF model + Ollama model multi-agent debate.")
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--hf_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ollama_model", default="llama3.1:8b")
    parser.add_argument("--hf_alias", default="qwen7")
    parser.add_argument("--ollama_alias", default="ollama_llama3_8b")
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    done = line_count(output_path) if args.resume else 0

    print(f"Loading HF model: {args.hf_model}")
    hf_model = load_hf_model(args.hf_model)
    start = time.time()

    for index in range(done, len(records)):
        row = records[index]
        per_start = time.time()
        error_text = ""
        try:
            debate = debate_one(
                row,
                hf_model,
                args.hf_alias,
                args.ollama_model,
                args.ollama_alias,
                args.ollama_host,
                args.max_new_tokens,
                args.timeout,
            )
        except Exception as exc:
            error_text = str(exc)
            debate = {
                f"{args.hf_alias}_1": "",
                f"{args.ollama_alias}_1": "",
                f"{args.hf_alias}_2": "",
                f"{args.ollama_alias}_2": "",
                f"{args.hf_alias}_final": "invalid",
                f"{args.ollama_alias}_final": "invalid",
            }
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
        print(
            f"[{index + 1}/{len(records)}] {args.hf_alias}={out[f'{args.hf_alias}_final']} "
            f"{args.ollama_alias}={out[f'{args.ollama_alias}_final']} "
            f"item_time={time.time() - per_start:.2f}s ETA={format_eta(start, index + 1, len(records))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
