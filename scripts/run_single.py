import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single-LLM experiment (single_model or self_reflection)."
    )
    parser.add_argument("--variant", choices=["single_model", "self_reflection"], required=True)
    parser.add_argument("--model", required=True, help="Model file name without .py (e.g., gemma, llama3)")
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
        "--prompt_type",
        choices=["without_rot", "with_rot"],
        default="without_rot",
        help="Prompt type for variant=single_model",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Also run evaluation after generation completes",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "single_llm" / args.variant / f"{args.model}.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        if args.variant == "single_model":
            output_path = repo_root / "results" / "single" / args.variant / f"{args.model}_{args.prompt_type}.jsonl"
        else:
            output_path = repo_root / "results" / "single" / args.variant / f"{args.model}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--input_path",
        str(args.input_path),
        "--output_path",
        str(output_path),
    ]

    if args.variant == "single_model":
        cmd.extend(["--type", args.prompt_type])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)

    if args.run_eval:
        eval_script = repo_root / "scripts" / "evaluate_outputs.py"
        eval_output = output_path.with_name(output_path.stem + "_accuracy.jsonl")
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--mode",
            "single",
            "--input_file",
            str(output_path),
            "--model",
            args.model,
            "--output_file",
            str(eval_output),
        ]
        print("Running:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=repo_root, check=True)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
