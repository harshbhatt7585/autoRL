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

# ── Redesigned blocky logo using full block / half-block characters ────────────
#    Each glyph is 7 chars wide × 5 rows tall, drawn with █ ▀ ▄ ▌ ▐ ░
LOGO_GLYPHS = {
    "A": [
        "  ███  ",
        " █   █ ",
        "███████",
        "█     █",
        "█     █",
    ],
    "U": [
        "█     █",
        "█     █",
        "█     █",
        "█     █",
        " █████ ",
    ],
    "T": [
        "███████",
        "  ███  ",
        "  ███  ",
        "  ███  ",
        "  ███  ",
    ],
    "O": [
        " █████ ",
        "█     █",
        "█     █",
        "█     █",
        " █████ ",
    ],
    "R": [
        "██████ ",
        "█     █",
        "██████ ",
        "█   █  ",
        "█    ██",
    ],
    "L": [
        "█      ",
        "█      ",
        "█      ",
        "█      ",
        "███████",
    ],
}

# Tagline shown beneath the logo
TAGLINE = "unattended  codex  experiment  loops"

# Box-drawing characters
_TL, _TR, _BL, _BR = "╔", "╗", "╚", "╝"
_H, _V = "═", "║"
_ML, _MR = "╠", "╣"          # mid-row left / right connectors


class Style:
    reset  = "\033[0m"
    bold   = "\033[1m"
    dim    = "\033[2m"
    italic = "\033[3m"
    # foreground
    cyan   = "\033[96m"
    white  = "\033[97m"
    green  = "\033[92m"
    yellow = "\033[93m"
    red    = "\033[91m"
    grey   = "\033[37m"
    # background accents
    bg_dark = "\033[48;5;235m"


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
    return parser.parse_args()


# ── colour helpers ────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("TERM", "") != "dumb"


def _c(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    return "".join(codes) + text + Style.reset


# ── header / banner ───────────────────────────────────────────────────────────

def _print_header() -> None:
    word   = "AUTORL"
    glyphs = [LOGO_GLYPHS[ch] for ch in word]
    height = len(glyphs[0])

    # assemble rows: glyphs separated by two spaces
    logo_rows: list[str] = []
    for row in range(height):
        logo_rows.append("   ".join(g[row] for g in glyphs))

    inner_w = max(len(r) for r in logo_rows)
    inner_w = max(inner_w, len(TAGLINE) + 2)   # ensure tagline fits
    box_w   = inner_w + 4                       # 2 side padding + 2 border chars

    border_top    = _TL + _H * (box_w - 2) + _TR
    border_bottom = _BL + _H * (box_w - 2) + _BR
    divider       = _ML + _H * (box_w - 2) + _MR

    print()
    print(_c(border_top, Style.cyan, Style.bold))
    # empty padding row
    print(_c(_V + " " * (box_w - 2) + _V, Style.cyan, Style.bold))
    for row in logo_rows:
        padding = box_w - 2 - len(row)
        lpad    = padding // 2
        rpad    = padding - lpad
        line    = _V + " " * lpad + row + " " * rpad + _V
        print(_c(line, Style.cyan, Style.bold))
    print(_c(_V + " " * (box_w - 2) + _V, Style.cyan, Style.bold))
    print(_c(divider, Style.cyan, Style.bold))
    # tagline row
    tpad   = box_w - 2 - len(TAGLINE)
    tlpad  = tpad // 2
    trpad  = tpad - tlpad
    tline  = _V + " " * tlpad + TAGLINE + " " * trpad + _V
    print(_c(tline, Style.grey, Style.italic))
    print(_c(border_bottom, Style.cyan, Style.bold))
    print()


def _print_kv(label: str, value: str) -> None:
    bullet = _c("▸", Style.cyan, Style.bold)
    key    = _c(label, Style.white, Style.bold)
    print(f"  {bullet} {key}  {value}")


# ── process helpers ───────────────────────────────────────────────────────────

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

    repo_q   = shlex.quote(str(repo))
    prompt_q = shlex.quote(prompt)
    log_q    = shlex.quote(str(log_path))
    pid_q    = shlex.quote(str(pid_path))
    loop_script = (
        "while true; do "
        f"codex -a never -s workspace-write exec -C {repo_q} {prompt_q}; "
        f"sleep {sleep_seconds}; "
        "done"
    )
    loop_script_q = shlex.quote(loop_script)
    return f"nohup bash -lc {loop_script_q} > {log_q} 2>&1 & echo $! > {pid_q}"


# ── plain log stream (non-tty fallback) ───────────────────────────────────────

def _stream_log(log_path: Path) -> None:
    print(
        _c("  ▸ Streaming log", Style.green, Style.bold)
        + _c(f"  {log_path}", Style.white)
        + _c("  (Ctrl+C stops viewing; loop keeps running)", Style.dim)
    )
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, 2)
        while True:
            line = handle.readline()
            if line:
                print(line, end="")
                continue
            time.sleep(0.2)


# ── TUI ───────────────────────────────────────────────────────────────────────

def _read_log_lines(log_path: Path, max_lines: int) -> list[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ["(log file not found yet)"]
    except OSError as exc:
        return [f"(failed reading log: {exc})"]
    lines = text.splitlines()
    return lines[-max_lines:] if len(lines) > max_lines else lines


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…" if width > 1 else text[:width]


def _run_tui(*, log_path: Path, pid_path: Path, repo_path: Path) -> None:

    def _draw(stdscr: curses.window) -> None:
        curses.start_color()
        curses.use_default_colors()
        # colour pairs
        curses.init_pair(1, curses.COLOR_CYAN,    -1)   # header / accents
        curses.init_pair(2, curses.COLOR_GREEN,   -1)   # running status
        curses.init_pair(3, curses.COLOR_YELLOW,  -1)   # warning / stopped
        curses.init_pair(4, curses.COLOR_RED,     -1)   # error
        curses.init_pair(5, curses.COLOR_WHITE,   -1)   # normal text
        curses.init_pair(6, curses.COLOR_BLACK,   curses.COLOR_CYAN)   # footer bar

        CY  = curses.color_pair(1)
        GR  = curses.color_pair(2)
        YL  = curses.color_pair(3)
        RD  = curses.color_pair(4)
        NM  = curses.color_pair(5)
        FT  = curses.color_pair(6)

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(250)

        status_text = "RUNNING"
        last_error  = ""

        SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spin_i  = 0

        while True:
            height, width = stdscr.getmaxyx()
            stdscr.erase()

            pid     = _read_running_pid(pid_path)
            running = pid is not None
            now     = datetime.now().strftime("%H:%M:%S")

            # ── title bar (row 0) ────────────────────────────────────────────
            spinner = SPINNER[spin_i % len(SPINNER)] if running else "■"
            spin_i += 1

            title   = f" AUTORL  {spinner}  {status_text}  "
            ts      = f"  {now} "
            gap     = width - len(title) - len(ts)
            bar     = title + " " * max(gap, 0) + ts
            try:
                stdscr.addstr(0, 0, _truncate(bar, width), CY | curses.A_BOLD | curses.A_REVERSE)
            except curses.error:
                pass

            # ── info rows (rows 1-4) ─────────────────────────────────────────
            def kv(row: int, label: str, val: str, val_attr: int = NM) -> None:
                try:
                    stdscr.addstr(row, 2, label, CY | curses.A_BOLD)
                    stdscr.addstr(row, 2 + len(label) + 1, _truncate(val, width - len(label) - 4), val_attr)
                except curses.error:
                    pass

            kv(1, "repo ▸", str(repo_path))
            kv(2, "log  ▸", str(log_path))
            pid_val  = str(pid) if pid else "stopped"
            pid_attr = GR | curses.A_BOLD if running else YL
            kv(3, "pid  ▸", pid_val, pid_attr)

            # ── divider ──────────────────────────────────────────────────────
            divider = "─" * (width - 2)
            try:
                stdscr.addstr(4, 1, _truncate(divider, width - 1), CY | curses.A_DIM)
                stdscr.addstr(4, 1, "▼ logs", CY | curses.A_BOLD)
            except curses.error:
                pass

            # ── log pane ─────────────────────────────────────────────────────
            log_top    = 5
            footer_row = height - 2
            log_height = max(1, footer_row - log_top)
            log_lines  = _read_log_lines(log_path, log_height)

            for idx, line in enumerate(log_lines):
                y = log_top + idx
                if y >= footer_row:
                    break
                try:
                    stdscr.addstr(y, 1, _truncate(line, width - 2), NM | curses.A_DIM)
                except curses.error:
                    pass

            # ── footer ───────────────────────────────────────────────────────
            controls = "  [q] quit view    [k] kill loop  "
            try:
                stdscr.addstr(footer_row, 0,
                              _truncate(controls.ljust(width), width), FT | curses.A_BOLD)
            except curses.error:
                pass

            if last_error:
                try:
                    stdscr.addstr(height - 1, 1,
                                  _truncate(f"⚠  {last_error}", width - 2), RD | curses.A_BOLD)
                except curses.error:
                    pass

            stdscr.refresh()

            # ── input ────────────────────────────────────────────────────────
            ch = stdscr.getch()
            if ch == -1:
                status_text = "RUNNING" if running else "STOPPED"
                continue
            if ch in (ord("q"), ord("Q")):
                break
            if ch in (ord("k"), ord("K")):
                if running and pid is not None:
                    try:
                        os.kill(pid, 15)
                        status_text = "STOPPING…"
                        last_error  = ""
                    except OSError as exc:
                        last_error = f"kill failed: {exc}"
                else:
                    last_error = "No running loop PID found."

    curses.wrapper(_draw)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    if not args.start:
        _print_header()
        print(_c("  Use  autorl --start  to launch the loop.\n", Style.yellow))
        return 2

    _print_header()

    repo_path = Path(args.repo).expanduser().resolve()
    log_path  = Path(args.log).expanduser()
    pid_path  = Path(args.pid_file).expanduser()

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
        print(_c("  ◆ Loop already running", Style.yellow, Style.bold))
        _print_kv("PID",     str(existing_pid))
        _print_kv("Log",     str(log_path))
        print()
        try:
            if not (sys.stdout.isatty() and sys.stdin.isatty()):
                _stream_log(log_path)
            else:
                _run_tui(log_path=log_path, pid_path=pid_path, repo_path=repo_path)
        except KeyboardInterrupt:
            print()
            print(_c("  Stopped log streaming. Background loop is still running.", Style.yellow))
        return 0

    print(_c("  ◆ Starting background loop…", Style.green, Style.bold))
    _print_kv("Repo",     str(repo_path))
    _print_kv("Log",      str(log_path))
    _print_kv("PID file", str(pid_path))
    _print_kv("Sleep",    f"{args.sleep}s")
    print()

    loop_command = _build_loop_command(
        repo=repo_path,
        prompt=prompt,
        log_path=log_path,
        pid_path=pid_path,
        sleep_seconds=args.sleep,
    )
    subprocess.run(["bash", "-lc", loop_command], check=True)

    pid_text = (
        pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "unknown"
    )
    print(_c("  ◆ autorl loop started", Style.green, Style.bold))
    _print_kv("PID", pid_text)
    print()

    try:
        if not (sys.stdout.isatty() and sys.stdin.isatty()):
            _stream_log(log_path)
        else:
            _run_tui(log_path=log_path, pid_path=pid_path, repo_path=repo_path)
    except KeyboardInterrupt:
        print()
        print(_c("  Stopped log streaming. Background loop is still running.", Style.yellow))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())