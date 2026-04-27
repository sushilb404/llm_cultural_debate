from __future__ import annotations

import argparse
import os
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def run(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> None:
    log("Running: " + " ".join(cmd), log_path)
    with log_path.open("a", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def wait_for_file(path: Path, expected_rows: int, poll_seconds: int, log_path: Path) -> None:
    while True:
        rows = line_count(path)
        log(f"Waiting for Qwen7+Gemma4 completion: {rows}/{expected_rows}", log_path)
        if rows >= expected_rows:
            return
        time.sleep(max(poll_seconds, 10))


def postprocess(repo_root: Path, args: argparse.Namespace, log_path: Path) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    run(
        [
            sys.executable,
            "scripts/postprocess_when_complete.py",
            "--target_file",
            args.target_file,
            "--expected_rows",
            str(args.expected_rows),
            "--poll_seconds",
            "10",
            "--first_model",
            "qwen7",
            "--second_model",
            "gemma4",
            "--analysis_dir",
            "results/analysis_qwen7_gemma4",
            "--drift_dir",
            "results/analysis_drift_qwen7_gemma4",
            "--figures_dir",
            "results/figures_qwen7_gemma4",
            "--ml_dir",
            "results/analysis_ml_qwen7_gemma4",
            "--integrity_dir",
            "results/analysis_integrity_qwen7_gemma4",
            "--sensitivity_dir",
            "results/analysis_sensitivity_qwen7_gemma4",
        ],
        repo_root,
        log_path,
        env=env,
    )


def push_proposal_branch(repo_root: Path, branch: str, log_path: Path) -> None:
    proposal_paths = [
        "results/multi/open_qwen7_gemma4_full.jsonl",
        "results/logs/qwen7_gemma4_resilient_runner.log",
        "results/analysis_qwen7_gemma4",
        "results/analysis_drift_qwen7_gemma4",
        "results/figures_qwen7_gemma4",
        "results/analysis_ml_qwen7_gemma4",
        "results/analysis_integrity_qwen7_gemma4",
        "results/analysis_sensitivity_qwen7_gemma4",
    ]

    run(["git", "switch", "-C", branch], repo_root, log_path)
    run(["git", "add", *proposal_paths], repo_root, log_path)

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if status.returncode == 0:
        log("No staged proposal changes to commit.", log_path)
    else:
        run(["git", "commit", "-m", "For the proposal: qwen7 gemma4 debate results"], repo_root, log_path)

    run(["git", "push", "-u", "origin", branch], repo_root, log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Qwen7+Gemma4 results for the proposal branch.")
    parser.add_argument("--target_file", default="results/multi/open_qwen7_gemma4_full.jsonl")
    parser.add_argument("--expected_rows", type=int, default=2633)
    parser.add_argument("--poll_seconds", type=int, default=180)
    parser.add_argument("--branch", default="for-the-proposal")
    parser.add_argument("--log_file", default="results/logs/for_the_proposal_finalizer.log")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    log_path = repo_root / args.log_file
    target_path = repo_root / args.target_file

    wait_for_file(target_path, args.expected_rows, args.poll_seconds, log_path)
    postprocess(repo_root, args, log_path)
    push_proposal_branch(repo_root, args.branch, log_path)
    log("For the proposal branch finalization complete.", log_path)


if __name__ == "__main__":
    main()
