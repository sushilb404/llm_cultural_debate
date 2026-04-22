import argparse
import json
import time
import sys
import re
from pathlib import Path
from typing import List
from datetime import datetime, timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_llm.single_model.prompt import prompts
from single_llm.single_model.utils import country_capitalized_mapping
from label_utils import normalize_label


OPEN_MODEL_DEFAULTS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

MODEL_ALIASES = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen7",
    "Qwen/Qwen2.5-3B-Instruct": "qwen3",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen15",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen05",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_alias(model_id: str) -> str:
    if model_id in MODEL_ALIASES:
        return MODEL_ALIASES[model_id]
    tail = model_id.split("/")[-1].lower()
    alias = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    return alias or "model"


def build_prompt(record: dict, use_rot: bool) -> str:
    country_small = record["Country"]
    country = country_capitalized_mapping[country_small]
    story = record["Story"]
    rot = record["Rule-of-Thumb"]
    if use_rot:
        return prompts["en_rot"].replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
    return prompts["en"].replace("{{country}}", country).replace("{{story}}", story)


def maybe_apply_chat_template(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def estimate_finish_time(start_time: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "unknown"
    elapsed = time.time() - start_time
    avg_per_item = elapsed / completed
    remaining = max(total - completed, 0) * avg_per_item
    finish = datetime.now() + timedelta(seconds=remaining)
    return finish.strftime("%Y-%m-%d %H:%M:%S")


def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    prompt = maybe_apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = "Answer (Yes, No or Neither):"
    if answer_start in text:
        generation = text.split(answer_start)[-1].strip()
    else:
        generation = text
    generation = generation.split("<")[0].strip()
    return generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a few open models on a small cultural benchmark subset.")
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--models", nargs="*", default=OPEN_MODEL_DEFAULTS)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--use_rot", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_path", default="results/pilot/open_model_compare.json")
    parser.add_argument(
        "--output_predictions_dir",
        default="",
        help="Optional directory to export per-model JSONL predictions for downstream benchmark scripts.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    records = list(read_jsonl(input_path))[: args.max_examples]
    if not records:
        raise ValueError("No records available to test.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_dir = Path(args.output_predictions_dir) if args.output_predictions_dir else None
    if predictions_dir:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    results = []
    run_start = time.time()
    for model_id in args.models:
        print(f"\n=== Testing {model_id} ===")
        alias = model_alias(model_id)
        try:
            tokenizer, model = load_model(model_id)
        except Exception as exc:
            print(f"Failed to load {model_id}: {exc}")
            results.append({
                "model_id": model_id,
                "status": "load_failed",
                "error": str(exc),
            })
            continue

        correct = 0
        total = 0
        durations: List[float] = []
        examples = []
        prediction_rows: List[dict] = []

        for idx, record in enumerate(records, start=1):
            prompt = build_prompt(record, args.use_rot)
            gold = normalize_label(record.get("Gold Label", ""))
            start = time.time()
            try:
                generation = generate_answer(tokenizer, model, prompt, args.max_new_tokens)
                elapsed = time.time() - start
                pred = normalize_label(generation)
            except Exception as exc:
                generation = ""
                pred = ""
                elapsed = time.time() - start
                examples.append({
                    "index": idx,
                    "gold": gold,
                    "prediction": pred,
                    "error": str(exc),
                    "elapsed_sec": round(elapsed, 4),
                })
                continue

            total += 1
            durations.append(elapsed)
            if pred == gold:
                correct += 1

            examples.append({
                "index": idx,
                "gold": gold,
                "prediction": pred,
                "elapsed_sec": round(elapsed, 4),
            })
            prediction_rows.append(
                {
                    "Country": record.get("Country", ""),
                    "Story": record.get("Story", ""),
                    "Rule-of-Thumb": record.get("Rule-of-Thumb", ""),
                    "Gold Label": record.get("Gold Label", ""),
                    alias: pred,
                    f"{alias}_final": pred,
                    "model_id": model_id,
                }
            )
            eta = estimate_finish_time(run_start, total, len(records))
            print(f"[{idx}/{len(records)}] gold={gold} pred={pred} time={elapsed:.2f}s ETA={eta}")

        accuracy = (correct / total) if total else 0.0
        avg_time = (sum(durations) / len(durations)) if durations else 0.0
        print(f"Accuracy on {total} valid examples: {accuracy:.3f}")
        print(f"Average generation time: {avg_time:.2f}s")

        results.append({
            "model_id": model_id,
            "alias": alias,
            "status": "ok",
            "max_examples": args.max_examples,
            "use_rot": args.use_rot,
            "valid_examples": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "average_generation_time_sec": round(avg_time, 4),
            "examples": examples,
        })

        if predictions_dir and prediction_rows:
            suffix = "with_rot" if args.use_rot else "without_rot"
            per_model_path = predictions_dir / f"{alias}_{suffix}.jsonl"
            write_jsonl(per_model_path, prediction_rows)
            print(f"Saved prediction JSONL: {per_model_path}")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved pilot comparison to: {output_path}")


if __name__ == "__main__":
    main()
