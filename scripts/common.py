"""Helpers shared by the command-line scripts: results files, summaries and plots.

Importing this module also puts the project root on ``sys.path``, so the
scripts can be run as ``python scripts/<name>.py`` from the project folder.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Columns that identify an episode or describe the controller's final state
# rather than measure performance; they are written to CSV but not summarised.
_NOT_SUMMARISED = {"episode", "seed", "update", "terminated", "green_phase", "time_in_green"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarise(records: list[dict]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of every numeric column, ignoring NaN."""
    summary = {}
    for key in dict.fromkeys(k for r in records for k in r):
        if key in _NOT_SUMMARISED:
            continue
        values = [float(r[key]) for r in records if key in r and not math.isnan(float(r[key]))]
        if not values:
            continue
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        summary[key] = (mean, std)
    return summary


def print_summary(title: str, records: list[dict]) -> None:
    print(f"\n{title} ({len(records)} episodes)")
    for key, (mean, std) in summarise(records).items():
        print(f"  {key:<24} {mean:12.3f}  +/- {std:.3f}")


def plot_training(episodes: list[dict], updates: list[dict], path: Path, title: str) -> None:
    """Save a figure with episode returns, critic loss and policy entropy."""
    import matplotlib
    matplotlib.use("Agg")  # write files without needing a display
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    returns = [e["return"] for e in episodes]
    window = max(1, min(10, len(returns)))
    moving = [sum(returns[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(returns))]
    axes[0].plot(range(1, len(returns) + 1), returns, alpha=0.35, label="episode")
    axes[0].plot(range(1, len(returns) + 1), moving, label=f"mean of last {window}")
    axes[0].set(xlabel="episode", ylabel="return", title="Training return")
    axes[0].legend()

    update_numbers = [u["update"] for u in updates]
    axes[1].plot(update_numbers, [u["value_loss"] for u in updates])
    axes[1].set(xlabel="update", ylabel="MSE (normalised returns)", title="Critic loss")
    axes[2].plot(update_numbers, [u["entropy"] for u in updates])
    axes[2].set(xlabel="update", ylabel="nats", title="Policy entropy")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
