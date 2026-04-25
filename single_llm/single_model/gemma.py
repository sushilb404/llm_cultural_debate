import argparse
import datetime
import jsonlines
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt import prompts
from scripts.label_utils import classify_label
from utils import country_capitalized_mapping


own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir


model_id = "google/gemma-4-E4B-it"


def _response_to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common response shapes: {"text": ...} or chat-style {"content": ...}
        for key in ("text", "content", "response", "answer"):
            if key in value:
                return _response_to_text(value[key])
        return " ".join(_response_to_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_response_to_text(item) for item in value)
    return str(value)

def main():
    start_time = datetime.datetime.now()

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--type", type=str, default="without_rot")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=own_cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        cache_dir=own_cache_dir,
        device_map="auto",
    )
    # =========================================== Load Dataset ===========================================    
    generations = []
    with jsonlines.open(args.output_path, mode="w") as outfile:
        with jsonlines.open(args.input_path) as file:
            for line in file.iter():
                country_small = line['Country']
                country = country_capitalized_mapping[country_small]
                story = line['Story']
                rot = line["Rule-of-Thumb"]
                
                if args.type == "without_rot":
                    en_prompt = prompts["en"]
                    prompt = en_prompt.replace("{{country}}", country).replace("{{story}}", story)
                elif args.type == "with_rot":
                    en_prompt = prompts["en_rot"]
                    prompt = en_prompt.replace("{{country}}", country).replace("{{story}}", story).replace("{{rot}}", rot)
                else:
                    raise ValueError("type argument should be either 'without_rot' or 'with_rot'.")
                print(prompt)
    
    # =========================================== Generation =============================================
                messages = [{"role": "user", "content": prompt}]
                prompt_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = processor(text=prompt_text, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                generated_text = processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
                if hasattr(processor, "parse_response"):
                    generated_text = processor.parse_response(generated_text)
                generation = classify_label(_response_to_text(generated_text))

                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"gemma4"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
