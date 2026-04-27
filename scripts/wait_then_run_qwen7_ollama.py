from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def log(message: str, log_path: Path) -> None:
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for Qwen7+Gemma4, then run Qwen7+Ollama debate.")
    parser.add_argument("--wait_file", default="results/multi/open_qwen7_gemma4_full.jsonl")
    parser.add_argument("--expected_rows", type=int, default=2633)
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--output_path", default="results/multi/open_qwen7_ollama_llama3_8b_full.jsonl")
    parser.add_argument("--log_file", default="results/logs/qwen7_ollama_llama3_8b_waiter.log")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    wait_file = repo_root / args.wait_file
    log_path = repo_root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        rows = line_count(wait_file)
        log(f"Waiting for Qwen7+Gemma4: {rows}/{args.expected_rows}", log_path)
        if rows >= args.expected_rows:
            break
        time.sleep(max(args.poll_seconds, 10))

    cmd = [
        sys.executable,
        "scripts/run_multi_hf_ollama_debate.py",
        "--input_path",
        "data/normad.jsonl",
        "--output_path",
        args.output_path,
        "--hf_model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--ollama_model",
        "llama3.1:8b",
        "--hf_alias",
        "qwen7",
        "--ollama_alias",
        "ollama_llama3_8b",
        "--max_new_tokens",
        "48",
        "--resume",
    ]
    log("Starting Qwen7+Ollama debate: " + " ".join(cmd), log_path)
    with (repo_root / "results/logs/qwen7_ollama_llama3_8b_debate.log").open("a", encoding="utf-8") as run_log:
        subprocess.run(cmd, cwd=repo_root, stdout=run_log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
