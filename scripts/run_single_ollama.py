from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import jsonlines

from multi_llm.prompt import prompts
from multi_llm.utils import country_capitalized_mapping
from scripts.label_utils import classify_label


def completed_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with jsonlines.open(path) as existing:
        return sum(1 for _ in existing.iter())


def ollama_generate(model: str, prompt: str, host: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 64,
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
    return str(data.get("response", ""))


def ensure_model(model: str) -> None:
    default_exe = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
    ollama_exe = os.environ.get("OLLAMA_EXE") or str(default_exe if default_exe.exists() else "ollama")
    subprocess.run([ollama_exe, "pull", model], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-model NORMAD baseline through Ollama.")
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--output_path", default="results/single/single_model/ollama_llama3_8b_without_rot.jsonl")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--field_name", default="ollama_llama3_8b")
    parser.add_argument("--type", default="without_rot", choices=["without_rot", "with_rot"])
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pull:
        ensure_model(args.model)

    completed = completed_rows(output_path) if args.resume else 0
    if completed:
        print(f"Resuming Ollama {args.model} from {completed} completed rows.")

    mode = "a" if args.resume and output_path.exists() else "w"
    with jsonlines.open(output_path, mode=mode) as outfile:
        with jsonlines.open(input_path) as infile:
            for row_index, line in enumerate(infile.iter()):
                if row_index < completed:
                    continue

                country = country_capitalized_mapping[line["Country"]]
                story = line["Story"]
                rot = line["Rule-of-Thumb"]

                if args.type == "without_rot":
                    prompt = prompts["en"].replace("{{country}}", country).replace("{{story}}", story)
                else:
                    prompt = (
                        prompts["en_rot"]
                        .replace("{{country}}", country)
                        .replace("{{story}}", story)
                        .replace("{{rot}}", rot)
                    )

                for attempt in range(3):
                    try:
                        response = ollama_generate(args.model, prompt, args.host, args.timeout)
                        break
                    except URLError:
                        if attempt == 2:
                            raise
                        time.sleep(5)

                line[args.field_name] = classify_label(response)
                outfile.write(line)
                print(f"{row_index + 1}: {line[args.field_name]}", flush=True)


if __name__ == "__main__":
    main()
