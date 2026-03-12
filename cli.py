from __future__ import annotations

import argparse
import curses
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import textwrap


DEFAULT_SLEEP_SECONDS = 2
DEFAULT_LOG_PATH = "codex.out"
DEFAULT_PID_PATH = ".log/autorl.pid"
DEFAULT_PROMPT = (
    "Continue the experiment loop in program.md from the current repo state. "
    "Read results.tsv, keep working on candidate/env.py and candidate/train.py, "
    "run another experiment, update results.tsv, and do not stop after a single "
    "accepted run. Only stop if the repo is broken or the process is interrupted."
)


class Style:
    reset = "\033[0m"
    bold = "\033[1m"
    cyan = "\033[36m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    dim = "\033[2m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="autorl",
        description="Run an unattended Codex autoRL loop and stream codex.out live.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start (or attach to) the autorl background loop.",
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
        "--from-start",
        action="store_true",
        help="Stream log from beginning instead of tailing from the end.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable full-screen TUI and use plain log streaming.",
    )
    return parser.parse_args()


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("TERM", "") != "dumb"


def _color(text: str, tone: str) -> str:
    if not _supports_color():
        return text
    return f"{tone}{text}{Style.reset}"


def _print_header() -> None:
    print()
    print(_color("autorl", Style.bold + Style.cyan))
    print(_color("run unattended Codex experiment loops", Style.dim))
    print()


def _print_kv(label: str, value: str) -> None:
    print(f"{_color(label + ':', Style.bold)} {value}")


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_running_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    raw = pid_path.read_text(encoding="utf-8").strip()
    if not raw.isdigit():
        return None
    pid = int(raw)
    return pid if _is_pid_running(pid) else None


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


def _stream_log(log_path: Path, *, from_start: bool) -> None:
    print(
        _color("Streaming log", Style.green)
        + f" {log_path} "
        + _color("(Ctrl+C stops viewing; loop keeps running)", Style.dim)
    )
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        if not from_start:
            handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                print(line, end="")
                continue
            time.sleep(0.2)


def _read_log_lines(log_path: Path, max_lines: int, *, from_start: bool) -> list[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ["(log file not found yet)"]
    except OSError as exc:
        return [f"(failed reading log: {exc})"]

    lines = text.splitlines()
    if from_start:
        return lines[:max_lines]
    if len(lines) <= max_lines:
        return lines
    return lines[-max_lines:]


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _run_tui(
    *,
    log_path: Path,
    pid_path: Path,
    repo_path: Path,
    from_start: bool,
) -> None:
    def _draw(stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(250)

        status_line = "Running"
        last_error = ""

        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()

            pid = _read_running_pid(pid_path)
            running = pid is not None
            now = datetime.now().strftime("%H:%M:%S")
            header = f"autorl | {status_line} | {now}"
            stdscr.addstr(0, 0, _truncate(header, width), curses.A_BOLD)

            stdscr.addstr(1, 0, _truncate(f"Repo: {repo_path}", width))
            stdscr.addstr(2, 0, _truncate(f"PID: {pid if pid is not None else 'stopped'}", width))
            stdscr.addstr(3, 0, _truncate(f"Log: {log_path}", width))

            log_top = 5
            footer_lines = 2
            log_height = max(4, height - log_top - footer_lines)
            log_lines = _read_log_lines(log_path, log_height, from_start=from_start)

            stdscr.addstr(log_top - 1, 0, _truncate("Logs", width), curses.A_UNDERLINE)
            for idx, line in enumerate(log_lines[:log_height]):
                stdscr.addstr(log_top + idx, 0, _truncate(line, width))

            footer = "Controls: q quit view | k kill loop"
            stdscr.addstr(height - 2, 0, _truncate(footer, width), curses.A_REVERSE)
            if last_error:
                stdscr.addstr(height - 1, 0, _truncate(last_error, width), curses.A_BOLD)

            stdscr.refresh()

            ch = stdscr.getch()
            if ch == -1:
                status_line = "Running" if running else "Stopped"
                continue
            if ch in (ord("q"), ord("Q")):
                break
            if ch in (ord("k"), ord("K")):
                if running and pid is not None:
                    try:
                        os.kill(pid, 15)
                        status_line = "Stopping"
                        last_error = ""
                    except OSError as exc:
                        last_error = f"Kill failed: {exc}"
                else:
                    last_error = "No running loop PID found."

    curses.wrapper(_draw)


def main() -> int:
    args = parse_args()

    if not args.start:
        print("Use `autorl --start` to launch the loop.")
        return 2

    _print_header()

    repo_path = Path(args.repo).expanduser().resolve()
    log_path = Path(args.log).expanduser()
    pid_path = Path(args.pid_file).expanduser()

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    prompt = DEFAULT_PROMPT

    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    existing_pid = _read_running_pid(pid_path)
    if existing_pid is not None:
        print(_color("autorl loop already running.", Style.yellow))
        _print_kv("PID", str(existing_pid))
        _print_kv("Log", str(log_path))
        try:
            if args.plain or not (sys.stdout.isatty() and sys.stdin.isatty()):
                _stream_log(log_path, from_start=args.from_start)
            else:
                _run_tui(
                    log_path=log_path,
                    pid_path=pid_path,
                    repo_path=repo_path,
                    from_start=args.from_start,
                )
        except KeyboardInterrupt:
            print()
            print(_color("Stopped log streaming. Background loop is still running.", Style.yellow))
        return 0

    print(_color("Starting background loop...", Style.green))
    _print_kv("Repo", str(repo_path))
    _print_kv("Log", str(log_path))
    _print_kv("PID file", str(pid_path))
    _print_kv("Sleep", str(args.sleep))
    print()

    loop_command = _build_loop_command(
        repo=repo_path,
        prompt=prompt,
        log_path=log_path,
        pid_path=pid_path,
        sleep_seconds=args.sleep,
    )
    subprocess.run(["bash", "-lc", loop_command], check=True)

    pid_text = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "unknown"
    print(_color("Started autorl background loop.", Style.green))
    _print_kv("PID", pid_text)
    print()

    try:
        if args.plain or not (sys.stdout.isatty() and sys.stdin.isatty()):
            _stream_log(log_path, from_start=args.from_start)
        else:
            _run_tui(
                log_path=log_path,
                pid_path=pid_path,
                repo_path=repo_path,
                from_start=args.from_start,
            )
    except KeyboardInterrupt:
        print()
        print(_color("Stopped log streaming. Background loop is still running.", Style.yellow))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
