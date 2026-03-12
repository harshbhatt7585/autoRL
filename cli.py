from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import time


DEFAULT_SLEEP_SECONDS = 2
DEFAULT_LOG_PATH = "codex.out"
DEFAULT_PID_PATH = ".log/autorl.pid"
DEFAULT_PROMPT = (
    "Continue the experiment loop in program.md from the current repo state. "
    "Read results.tsv, keep working on candidate/env.py and candidate/train.py, "
    "run another experiment, update results.tsv, and do not stop after a single "
    "accepted run. Only stop if the repo is broken or the process is interrupted."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autorl",
        description="Run an unattended Codex autoRL loop and stream codex.out live.",
    )
    parser.add_argument(
        "--repo",
        default=str(Path.cwd()),
        help="Repository path passed to `codex exec -C` (default: current directory).",
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Log file written by nohup (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--pid-file",
        default=DEFAULT_PID_PATH,
        help=f"PID file for the background loop (default: {DEFAULT_PID_PATH}).",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Sleep seconds between codex loop iterations (default: {DEFAULT_SLEEP_SECONDS}).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt for codex exec. If omitted, prompts interactively.",
    )
    return parser.parse_args()


def _resolve_prompt(arg_prompt: str | None) -> str:
    if arg_prompt is not None:
        prompt = arg_prompt.strip()
        if not prompt:
            raise ValueError("--prompt cannot be empty.")
        return prompt

    print("Enter prompt for autorl. Press Enter on empty line to use default.")
    user_prompt = input("Prompt> ").strip()
    return user_prompt if user_prompt else DEFAULT_PROMPT


def _build_loop_command(
    *,
    repo: Path,
    prompt: str,
    log_path: Path,
    pid_path: Path,
    sleep_seconds: int,
) -> str:
    if sleep_seconds < 1:
        raise ValueError("--sleep must be at least 1.")

    repo_q = shlex.quote(str(repo))
    prompt_q = shlex.quote(prompt)
    log_q = shlex.quote(str(log_path))
    pid_q = shlex.quote(str(pid_path))

    return (
        "nohup bash -lc '"
        "while true; do "
        f"codex -a never -s workspace-write exec -C {repo_q} {prompt_q}; "
        f"sleep {sleep_seconds}; "
        "done"
        f"' > {log_q} 2>&1 & echo $! > {pid_q}"
    )


def _stream_log(log_path: Path) -> None:
    print(f"Streaming {log_path} (Ctrl+C to stop streaming; loop keeps running).")
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                print(line, end="")
                continue
            time.sleep(0.2)


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo).expanduser().resolve()
    log_path = Path(args.log).expanduser()
    pid_path = Path(args.pid_file).expanduser()

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    prompt = _resolve_prompt(args.prompt)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    loop_command = _build_loop_command(
        repo=repo_path,
        prompt=prompt,
        log_path=log_path,
        pid_path=pid_path,
        sleep_seconds=args.sleep,
    )
    subprocess.run(["bash", "-lc", loop_command], check=True)

    pid_text = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "unknown"
    print(f"Started autorl background loop with PID {pid_text}.")

    try:
        _stream_log(log_path)
    except KeyboardInterrupt:
        print("\nStopped log streaming. Background loop is still running.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
