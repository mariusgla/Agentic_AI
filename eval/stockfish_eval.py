"""
Move quality: how much does each move lose according to Stockfish, by move number?

Scores the model's legal top moves and, as a reference, the moves the humans played.
"""
import chess
import chess.engine
import matplotlib
matplotlib.use("Agg")       # only save files, never open windows (window code crashes next to Stockfish's thread)
import matplotlib.pyplot as plt
import pandas as pd

from common import BUCKET_SIZE, GRID_COLOR, HUMAN_COLOR, LINE_COLOR, bucket_start, iter_positions, token_to_legal_move

STOCKFISH = "/usr/games/stockfish"
NODES = 10_000      # search effort per position, fixed so that runs are comparable
MATE_SCORE = 10_000
MAX_LOSS = 1000     # one huge blunder should not dominate the results
MIN_MOVES = 20      # groups with fewer moves are too noisy to plot


def position_score(engine, board, player):
    """Stockfish's score for this position in centipawns, from the player's point of view."""
    # A new game object makes Stockfish forget earlier searches. Otherwise a score would depend
    # on which positions were searched before it, so the same human move could score differently
    # depending on which model is being evaluated.
    info = engine.analyse(board, chess.engine.Limit(nodes=NODES), game=object())
    return info["score"].pov(player).score(mate_score=MATE_SCORE)


def move_loss(engine, board, move, score_before):
    """How many centipawns the move gives away compared to the score before it."""
    board_after = board.copy()
    board_after.push(move)
    score_after = position_score(engine, board_after, board.turn)

    loss = score_before - score_after
    loss = max(loss, 0)         # search noise can make a move look slightly better than the best
    loss = min(loss, MAX_LOSS)
    return loss


def score(games, predictions, itos):
    """Returns one row per scored move: the human's move, and the model's if it was legal."""
    rows = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH) as engine:
        engine.configure({"Threads": 1})

        for game_id, half_move, board, human_move in iter_positions(games, itos):
            score_before = position_score(engine, board, board.turn)

            moves_to_score = {"human": human_move}
            token = int(predictions[game_id][half_move])
            model_move = token_to_legal_move(token, board, itos)
            if model_move is not None:
                moves_to_score["model"] = model_move

            for player, move in moves_to_score.items():
                rows.append({
                    "game_id": game_id,
                    "half_move": half_move,
                    "player": player,
                    "move": move.uci(),
                    "cp_loss": move_loss(engine, board, move, score_before),
                })
    return pd.DataFrame(rows)


def summarize(results):
    """Returns the median loss per player and group of half-moves, leaving out groups with too few moves."""
    groups = results.groupby(["player", bucket_start(results["half_move"])])

    rows = []
    for (player, first_half_move), group in groups:
        if len(group) >= MIN_MOVES:
            rows.append({
                "player": player,
                "first_half_move": first_half_move,
                "moves": len(group),
                "median_cp_loss": group["cp_loss"].median(),
            })
    return pd.DataFrame(rows)


def plot(summary, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for player, color in (("model", LINE_COLOR), ("human", HUMAN_COLOR)):
        player_rows = summary[summary["player"] == player]
        ax.plot(player_rows["first_half_move"], player_rows["median_cp_loss"], color=color, label=player,
                marker="o", markersize=8, linewidth=2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"half-move (start of each {BUCKET_SIZE}-half-move group)")
    ax.set_ylabel("median centipawn loss (lower is better)")
    ax.set_title(title)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
