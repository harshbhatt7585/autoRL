from __future__ import annotations

import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results.tsv")
OUTPUT_PATH = Path("results_score_graph.png")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if row.get("commit", "").strip()]
    return rows


def plot_experiment_scores(
    *,
    results_path: Path = RESULTS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    rows = load_rows(results_path)
    if not rows:
        raise RuntimeError("results.tsv has no experiment rows.")

    x = list(range(1, len(rows) + 1))
    y = [float(row["score"]) for row in rows]
    labels = [row["description"].strip() or "no description" for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10), dpi=160)

    ax.plot(x, y, color="#0A66C2", linewidth=1.2, alpha=0.65, zorder=1)
    ax.scatter(x, y, s=42, color="#0A66C2", edgecolor="white", linewidth=0.7, zorder=2)

    for index, (exp_id, score, label) in enumerate(zip(x, y, labels), start=1):
        text = textwrap.fill(label, width=28)
        offset = 16 if index % 2 else -18
        va = "bottom" if offset > 0 else "top"
        ax.annotate(
            text,
            xy=(exp_id, score),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.20",
                "fc": "white",
                "ec": "#C6CCD5",
                "alpha": 0.92,
            },
            zorder=3,
        )

    ax.set_title("Agent Performance by Experiment", fontsize=14, pad=14)
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xlim(0.5, len(rows) + 0.5)
    y_min = min(y)
    y_max = max(y)
    margin = max(0.08, (y_max - y_min) * 0.12)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> int:
    output_path = plot_experiment_scores()
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
