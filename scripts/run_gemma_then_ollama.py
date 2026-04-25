from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\nRunning: " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 4 E4B, then an Ollama 8B model, then evaluate both.")
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--type", default="without_rot", choices=["without_rot", "with_rot"])
    parser.add_argument("--ollama_model", default="llama3.1:8b")
    parser.add_argument("--ollama_field", default="ollama_llama3_8b")
    parser.add_argument("--pull_ollama", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "results" / "single" / "single_model"
    log_dir = repo_root / "results" / "logs"
    input_path = repo_root / args.input_path

    gemma_output = output_dir / f"gemma4_e4b_{args.type}.jsonl"
    ollama_output = output_dir / f"{args.ollama_field}_{args.type}.jsonl"

    run(
        [
            sys.executable,
            "single_llm/single_model/gemma.py",
            "--input_path",
            str(input_path),
            "--output_path",
            str(gemma_output),
            "--type",
            args.type,
            "--resume",
        ],
        repo_root,
        log_dir / "gemma4_then_ollama.log",
    )

    ollama_cmd = [
        sys.executable,
        "scripts/run_single_ollama.py",
        "--input_path",
        str(input_path),
        "--output_path",
        str(ollama_output),
        "--model",
        args.ollama_model,
        "--field_name",
        args.ollama_field,
        "--type",
        args.type,
        "--resume",
    ]
    if args.pull_ollama:
        ollama_cmd.append("--pull")
    run(ollama_cmd, repo_root, log_dir / "gemma4_then_ollama.log")

    for model, output in (("gemma4", gemma_output), (args.ollama_field, ollama_output)):
        run(
            [
                sys.executable,
                "scripts/evaluate_outputs.py",
                "--mode",
                "single",
                "--input_file",
                str(output),
                "--model",
                model,
            ],
            repo_root,
            log_dir / "gemma4_then_ollama.log",
        )


if __name__ == "__main__":
    main()
