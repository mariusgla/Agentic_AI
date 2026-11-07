"""
Legal move rate: how often is the model's top move legal, by move number?
"""
import matplotlib
matplotlib.use("Agg")       # only save files, never open windows (window code crashes next to Stockfish's thread)
import matplotlib.pyplot as plt
import pandas as pd

from common import GRID_COLOR, LINE_COLOR, bucket_start, iter_positions, token_to_legal_move


def score(games, predictions, itos):
    """Returns one row per position: was the model's top move legal?"""
    rows = []
    for game_id, half_move, board, _ in iter_positions(games, itos):
        token = int(predictions[game_id][half_move])
        is_legal = token_to_legal_move(token, board, itos) is not None
        rows.append({"half_move": half_move, "legal": is_legal})
    return pd.DataFrame(rows)


def summarize(positions):
    """Returns one row per group of half-moves: how many positions, and how many were legal."""
    groups = positions.groupby(bucket_start(positions["half_move"]))

    rows = []
    for first_half_move, group in groups:
        rows.append({
            "first_half_move": first_half_move,
            "last_half_move": group["half_move"].max() + 1,     # half_move counts from 0
            "positions": len(group),
            "legal_pct": group["legal"].mean() * 100,
        })
    return pd.DataFrame(rows)


def plot(summary, title, path):
    labels = []
    for first, last in zip(summary["first_half_move"], summary["last_half_move"]):
        labels.append(f"{first}-{last}")
    percentages = list(summary["legal_pct"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(labels, percentages, color=LINE_COLOR, marker="o", markersize=8, linewidth=2)

    # Write the value only on the first and last point, to keep the chart readable
    for i in (0, -1):
        ax.annotate(f"{percentages[i]:.1f}%", (labels[i], percentages[i]),
                    xytext=(0, 10), textcoords="offset points", ha="center")

    ax.set_ylim(0, 105)
    ax.set_xlabel("half-move")
    ax.set_ylabel("top move is legal (%)")
    ax.set_title(title)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
