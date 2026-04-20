import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-LLM debate experiment."
    )
    parser.add_argument(
        "--pair",
        default="",
        help="Pair script name without .py (e.g., gemma_aya). If omitted, --first_model and --second_model are required.",
    )
    parser.add_argument("--first_model", default="", help="First model in pair (used when --pair is omitted)")
    parser.add_argument("--second_model", default="", help="Second model in pair (used when --pair is omitted)")
    parser.add_argument(
        "--input_path",
        default="data/normad.jsonl",
        help="Input JSONL path (default: data/normad.jsonl)",
    )
    parser.add_argument(
        "--output_path",
        default="",
        help="Optional output file path. If omitted, a default path is generated under results/.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Also run evaluation after generation completes",
    )
    args = parser.parse_args()

    if args.pair:
        pair = args.pair
    else:
        if not args.first_model or not args.second_model:
            raise ValueError("Provide either --pair or both --first_model and --second_model.")
        pair = f"{args.first_model}_{args.second_model}"

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "multi_llm" / f"{pair}.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = repo_root / "results" / "multi" / f"{pair}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--input_path",
        str(args.input_path),
        "--output_path",
        str(output_path),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)

    if args.run_eval:
        if "_" not in pair:
            raise ValueError("Pair must be in first_second format for evaluation.")

        first_model, second_model = pair.split("_", 1)
        eval_script = repo_root / "scripts" / "evaluate_outputs.py"
        eval_output = output_path.with_name(output_path.stem + "_accuracy.jsonl")
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--mode",
            "multi",
            "--input_file",
            str(output_path),
            "--first_model",
            first_model,
            "--second_model",
            second_model,
            "--output_file",
            str(eval_output),
        ]
        print("Running:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=repo_root, check=True)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
