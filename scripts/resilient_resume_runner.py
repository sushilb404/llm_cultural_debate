import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


NON_RETRIABLE_CONFIG_ERROR_EXIT_CODE = 2


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_path: Optional[Path]) -> None:
    text = f"[{now_str()}] {message}"
    print(text)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def build_worker_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "scripts/run_multi_open_debate.py",
        "--input_path",
        args.input_path,
        "--output_path",
        args.output_path,
        "--model_a",
        args.model_a,
        "--model_b",
        args.model_b,
        "--alias_a",
        args.alias_a,
        "--alias_b",
        args.alias_b,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--resume",
    ]


def start_worker(cmd: List[str], cwd: Path, log_path: Optional[Path]) -> subprocess.Popen:
    log(f"Starting worker: {' '.join(cmd)}", log_path)
    proc = subprocess.Popen(cmd, cwd=cwd)
    log(f"Worker started with PID={proc.pid}", log_path)
    return proc


def stop_worker(proc: subprocess.Popen, log_path: Optional[Path]) -> None:
    if proc.poll() is not None:
        return
    log(f"Stopping worker PID={proc.pid}", log_path)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        log(f"Force-killing worker PID={proc.pid}", log_path)
        proc.kill()


def should_restart_worker(exit_code: int) -> bool:
    return exit_code != NON_RETRIABLE_CONFIG_ERROR_EXIT_CODE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run debate generation with automatic crash/stall recovery using --resume."
    )
    parser.add_argument("--input_path", default="data/normad.jsonl")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_rows", type=int, required=True)
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--alias_a", required=True)
    parser.add_argument("--alias_b", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--poll_seconds", type=int, default=90)
    parser.add_argument(
        "--stall_seconds",
        type=int,
        default=1800,
        help="Restart worker if output row count does not increase for this many seconds.",
    )
    parser.add_argument(
        "--log_file",
        default="",
        help="Optional log file path for supervisor events.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_path = (repo_root / args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file).resolve() if args.log_file else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_worker_command(args)

    # Track output growth and restart on crashes or long stalls.
    last_rows = line_count(output_path)
    last_growth_time = time.time()
    log(
        f"Supervisor started. Progress {last_rows}/{args.expected_rows}. Output file: {output_path}",
        log_path,
    )

    worker = start_worker(cmd, repo_root, log_path)

    while True:
        time.sleep(max(args.poll_seconds, 10))

        rows = line_count(output_path)
        if rows > last_rows:
            last_rows = rows
            last_growth_time = time.time()
            log(f"Progress advanced: {rows}/{args.expected_rows}", log_path)

        if rows >= args.expected_rows:
            log("Expected row count reached. Supervisor exiting.", log_path)
            stop_worker(worker, log_path)
            break

        exit_code = worker.poll()
        if exit_code is not None:
            if should_restart_worker(exit_code):
                log(f"Worker exited with code {exit_code}. Restarting with --resume.", log_path)
                worker = start_worker(cmd, repo_root, log_path)
            else:
                log(
                    f"Worker exited with non-retriable configuration error code {exit_code}. "
                    "Supervisor exiting.",
                    log_path,
                )
                raise SystemExit(exit_code)
            continue

        stalled_for = time.time() - last_growth_time
        if stalled_for >= args.stall_seconds:
            log(
                f"No progress for {int(stalled_for)}s (threshold={args.stall_seconds}s). Restarting worker.",
                log_path,
            )
            stop_worker(worker, log_path)
            worker = start_worker(cmd, repo_root, log_path)
            # Give restarted worker a grace period.
            last_growth_time = time.time()


if __name__ == "__main__":
    main()
