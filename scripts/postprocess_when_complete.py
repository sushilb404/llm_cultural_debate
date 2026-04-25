import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def dedupe_jsonl_inplace(path: Path) -> tuple[int, int]:
    """Deduplicate rows in-place by core scenario key.

    Returns (rows_before, rows_after).
    """
    if not path.exists():
        return 0, 0

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    before = len(rows)
    seen = set()
    deduped = []
    for row in rows:
        key = (
            row.get("Country", ""),
            row.get("Story", ""),
            row.get("Rule-of-Thumb", ""),
            row.get("Gold Label", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    after = len(deduped)
    if after != before:
        backup = path.with_suffix(path.suffix + ".pre_dedupe.bak")
        if not backup.exists():
            path.replace(backup)
        else:
            # If backup already exists from prior runs, overwrite target directly.
            path.unlink(missing_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for row in deduped:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return before, after


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a JSONL run to finish, then execute post-processing.")
    parser.add_argument("--target_file", required=True)
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--first_model", required=True)
    parser.add_argument("--second_model", required=True)
    parser.add_argument("--analysis_dir", default="results/analysis_qwen7")
    parser.add_argument("--drift_dir", default="results/analysis_drift_qwen7")
    parser.add_argument("--figures_dir", default="results/figures_qwen7")
    parser.add_argument("--ml_dir", default="results/analysis_ml_qwen7")
    parser.add_argument("--integrity_dir", default="results/analysis_integrity_qwen7")
    parser.add_argument("--sensitivity_dir", default="results/analysis_sensitivity_qwen7")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    target_file = repo_root / args.target_file

    print(f"Waiting for completion: {target_file}")
    while True:
        current = line_count(target_file)
        print(f"Progress lines: {current}/{args.expected_rows}")
        if current >= args.expected_rows:
            before, after = dedupe_jsonl_inplace(target_file)
            if after != before:
                print(f"Deduplicated rows: {before} -> {after}")

            if after >= args.expected_rows:
                print("Target unique row count reached. Starting post-processing pipeline...")
                break

            print(
                f"Unique rows after dedupe are {after}/{args.expected_rows}. "
                "Continuing to wait for additional completed rows..."
            )
        time.sleep(max(args.poll_seconds, 10))

    py = sys.executable

    # 1) Standard multi-model accuracy evaluation.
    run_cmd(
        [
            py,
            "scripts/evaluate_outputs.py",
            "--mode",
            "multi",
            "--input_file",
            args.target_file,
            "--first_model",
            args.first_model,
            "--second_model",
            args.second_model,
            "--output_file",
            f"{args.analysis_dir}/multi_accuracy.jsonl",
        ],
        repo_root,
    )

    # 1b) Integrity checks to guard duplicate/leakage artifacts.
    run_cmd(
        [
            py,
            "scripts/validate_results_integrity.py",
            "--input_file",
            args.target_file,
            "--reference_file",
            "data/normad.jsonl",
            "--expected_rows",
            str(args.expected_rows),
            "--output_json",
            f"{args.integrity_dir}/integrity_report.json",
            "--duplicates_csv",
            f"{args.integrity_dir}/duplicate_rows.csv",
        ],
        repo_root,
    )

    # 2) Fairness benchmark for each debater final output.
    run_cmd(
        [
            py,
            "scripts/benchmark_cultural.py",
            "--input_file",
            args.target_file,
            "--mode",
            "multi",
            "--model",
            args.first_model,
            "--output_json",
            f"{args.analysis_dir}/{args.first_model}_benchmark.json",
            "--output_country_csv",
            f"{args.analysis_dir}/{args.first_model}_country_metrics.csv",
        ],
        repo_root,
    )

    # 2b) Significance tests for model-vs-model headline differences.
    run_cmd(
        [
            py,
            "scripts/significance_report.py",
            "--input_file",
            args.target_file,
            "--first_model",
            args.first_model,
            "--second_model",
            args.second_model,
            "--output_json",
            f"{args.analysis_dir}/significance_report.json",
        ],
        repo_root,
    )
    run_cmd(
        [
            py,
            "scripts/benchmark_cultural.py",
            "--input_file",
            args.target_file,
            "--mode",
            "multi",
            "--model",
            args.second_model,
            "--output_json",
            f"{args.analysis_dir}/{args.second_model}_benchmark.json",
            "--output_country_csv",
            f"{args.analysis_dir}/{args.second_model}_country_metrics.csv",
        ],
        repo_root,
    )

    # 3) Drift metrics and visuals.
    run_cmd(
        [
            py,
            "scripts/analyze_drift.py",
            "--input_file",
            args.target_file,
            "--output_dir",
            args.drift_dir,
            "--models",
            args.first_model,
            args.second_model,
        ],
        repo_root,
    )
    run_cmd(
        [
            py,
            "scripts/plot_benchmark_results.py",
            "--analysis_dir",
            args.drift_dir,
            "--output_dir",
            args.figures_dir,
        ],
        repo_root,
    )

    # 4) K-means + Random Forest conversation analysis.
    run_cmd(
        [
            py,
            "scripts/analyze_conversation_ml.py",
            "--input_file",
            args.target_file,
            "--output_dir",
            args.ml_dir,
            "--models",
            args.first_model,
            args.second_model,
            "--k",
            "4",
        ],
        repo_root,
    )

    # 4b) AI-to-AI interaction load and agreement summary.
    run_cmd(
        [
            py,
            "scripts/analyze_ai_interactions.py",
            "--input_files",
            args.target_file,
            args.target_file,
            "--model_names",
            args.first_model,
            args.second_model,
            "--output_dir",
            f"{args.analysis_dir}/ai_interactions",
        ],
        repo_root,
    )

    # 5) Judge sensitivity check (parser strictness by default).
    run_cmd(
        [
            py,
            "scripts/judge_sensitivity_check.py",
            "--input_file",
            args.target_file,
            "--model",
            args.first_model,
            "--output_json",
            f"{args.sensitivity_dir}/{args.first_model}_judge_sensitivity.json",
            "--output_csv",
            f"{args.sensitivity_dir}/{args.first_model}_judge_disagreements.csv",
        ],
        repo_root,
    )
    run_cmd(
        [
            py,
            "scripts/judge_sensitivity_check.py",
            "--input_file",
            args.target_file,
            "--model",
            args.second_model,
            "--output_json",
            f"{args.sensitivity_dir}/{args.second_model}_judge_sensitivity.json",
            "--output_csv",
            f"{args.sensitivity_dir}/{args.second_model}_judge_disagreements.csv",
        ],
        repo_root,
    )

    print("Post-processing complete.")
    print(f"- analysis: {args.analysis_dir}")
    print(f"- drift: {args.drift_dir}")
    print(f"- figures: {args.figures_dir}")
    print(f"- ml: {args.ml_dir}")
    print(f"- integrity: {args.integrity_dir}")
    print(f"- sensitivity: {args.sensitivity_dir}")


if __name__ == "__main__":
    main()
