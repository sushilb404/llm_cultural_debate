import argparse
import datetime
import jsonlines
import os
import sys
from pathlib import Path

from transformers import pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt import prompts
from scripts.label_utils import classify_label
from utils import country_capitalized_mapping


model_id = "Qwen/Qwen2.5-7B-Instruct"


own_cache_dir = ""
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

def main():
    start_time = datetime.datetime.now()

    # =========================================== Parameter Setup ===========================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--type", type=str, default="without_rot")
    args = parser.parse_args()

    pipe = pipeline("text-generation", model=model_id, torch_dtype="auto", device_map="auto")

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

                outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
                generated_text = outputs[0]["generated_text"]

                generation = classify_label(generated_text)
                
                print(f"> {generation}")
                print("\n======================================================\n")
                generations.append(generation)

                line[f"qwen"] = generation
                outfile.write(line)
    
    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()