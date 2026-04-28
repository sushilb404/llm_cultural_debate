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


def run(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None, force_log: bool = True) -> None:
    log("Running: " + " ".join(cmd), log_path)
    with log_path.open("a", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=f if force_log else None, stderr=subprocess.STDOUT, env=env)


def wait_for_file(path: Path, expected_rows: int, poll_seconds: int, log_path: Path) -> None:
    while True:
        rows = line_count(path)
        log(f"Waiting for Qwen7+Ollama Llama3.1 8B completion: {rows}/{expected_rows}", log_path)
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
            "ollama_llama3_8b",
            "--analysis_dir",
            "results/analysis_qwen7_ollama_llama3_8b",
            "--drift_dir",
            "results/analysis_drift_qwen7_ollama_llama3_8b",
            "--figures_dir",
            "results/figures_qwen7_ollama_llama3_8b",
            "--ml_dir",
            "results/analysis_ml_qwen7_ollama_llama3_8b",
            "--integrity_dir",
            "results/analysis_integrity_qwen7_ollama_llama3_8b",
            "--sensitivity_dir",
            "results/analysis_sensitivity_qwen7_ollama_llama3_8b",
        ],
        repo_root,
        log_path,
        env=env,
    )
    run(
        [
            sys.executable,
            "scripts/analyze_cross_run_ml.py",
            "--output_dir",
            "results/analysis_cross_run_ml",
            "--k",
            "5",
        ],
        repo_root,
        log_path,
        env=env,
    )


def git_has_staged_changes(repo_root: Path) -> bool:
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return status.returncode != 0


def push_proposal_branch(repo_root: Path, branch: str, log_path: Path) -> None:
    proposal_paths = [
        "scripts/analyze_cross_run_ml.py",
        "scripts/finalize_qwen7_ollama_for_proposal.py",
        "results/multi/open_qwen7_ollama_llama3_8b_full.jsonl",
        "results/analysis_qwen7_ollama_llama3_8b",
        "results/analysis_drift_qwen7_ollama_llama3_8b",
        "results/figures_qwen7_ollama_llama3_8b",
        "results/analysis_ml_qwen7_ollama_llama3_8b",
        "results/analysis_integrity_qwen7_ollama_llama3_8b",
        "results/analysis_sensitivity_qwen7_ollama_llama3_8b",
        "results/analysis_cross_run_ml",
    ]
    log_paths = [
        "results/logs/qwen7_ollama_llama3_8b_debate.log",
        args_log_name(log_path, repo_root),
    ]

    run(["git", "switch", branch], repo_root, log_path)
    run(["git", "add", *proposal_paths], repo_root, log_path)
    existing_logs = [path for path in log_paths if (repo_root / path).exists()]
    if existing_logs:
        run(["git", "add", "-f", *existing_logs], repo_root, log_path)

    if git_has_staged_changes(repo_root):
        run(
            ["git", "commit", "-m", "For the proposal: qwen7 ollama debate results and cross-run ml"],
            repo_root,
            log_path,
        )
    else:
        log("No staged Qwen7+Ollama proposal changes to commit.", log_path)

    run(["git", "push", "-u", "origin", branch], repo_root, log_path)


def args_log_name(log_path: Path, repo_root: Path) -> str:
    try:
        return str(log_path.relative_to(repo_root))
    except ValueError:
        return str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Qwen7+Ollama Llama3.1 8B results for the proposal branch.")
    parser.add_argument("--target_file", default="results/multi/open_qwen7_ollama_llama3_8b_full.jsonl")
    parser.add_argument("--expected_rows", type=int, default=2633)
    parser.add_argument("--poll_seconds", type=int, default=180)
    parser.add_argument("--branch", default="for-the-proposal")
    parser.add_argument("--log_file", default="results/logs/qwen7_ollama_finalizer.log")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    log_path = repo_root / args.log_file
    target_path = repo_root / args.target_file

    wait_for_file(target_path, args.expected_rows, args.poll_seconds, log_path)
    postprocess(repo_root, args, log_path)
    push_proposal_branch(repo_root, args.branch, log_path)
    log("Qwen7+Ollama proposal finalization complete.", log_path)


if __name__ == "__main__":
    main()
