"""
Code shared by the eval: loading the model and games, predicting, replaying games.
"""
import os
import pickle
import sys

import chess
import numpy as np
import torch

# model.py is in the repo root, one folder above this file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from model import GPT, GPTConfig

BATCH_SIZE = 64
BUCKET_SIZE = 20    # results are grouped into blocks of this many half-moves

# Tokens 0, 1 and 2 are <pad>, <start> and <end>. Every token from 3 upwards is a chess move.
FIRST_MOVE_TOKEN = 3

LINE_COLOR = "#2a78d6"
HUMAN_COLOR = "#eb6834"
GRID_COLOR = "#e4e4e0"


def load_model(checkpoint_path):
    # weights_only=True refuses to run code hidden in a checkpoint, so other people's files are safe to load
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_args"]))

    # Compiled models save their weights with this prefix on every name
    weights = {}
    for name, value in checkpoint["model"].items():
        weights[name.removeprefix("_orig_mod.")] = value
    model.load_state_dict(weights)

    model.eval()
    return model


def load_games(data_dir, split):
    """Returns a table of games (one row per game) and the token -> move dictionary."""
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    tokens_per_game = meta["block_size"] + 1

    data = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16)
    games = data.reshape(-1, tokens_per_game).astype(np.int64)
    return games, meta["itos"]


@torch.no_grad()
def predict(model, games):
    """Returns the model's top token for every position: predictions[game_id][half_move]."""
    all_predictions = []
    for start in range(0, len(games), BATCH_SIZE):
        batch = torch.from_numpy(games[start:start + BATCH_SIZE])
        inputs = batch[:, :-1]
        targets = batch[:, 1:].contiguous()

        # We ignore the loss, but passing targets makes the model return
        # logits for every position in the game instead of only the last one.
        logits, _ = model(inputs, targets)

        top_tokens = logits.argmax(dim=-1)
        all_predictions.append(top_tokens.numpy())
    return np.concatenate(all_predictions)


def iter_positions(games, itos):
    """
    Replays every game and yields (game_id, half_move, board, human_move) before each move.

    half_move counts from 0, so it can be used directly as predictions[game_id][half_move].
    Always plays the human's move, so every model is tested on the same positions.
    The same board object is reused: call board.copy() if you want to keep it.
    """
    for game_id, game in enumerate(games):
        board = chess.Board()
        # Index 0 of a game is <start>, so the first move is at index 1
        for half_move, token in enumerate(game[1:]):
            token = int(token)
            if token < FIRST_MOVE_TOKEN:    # <pad> or <end>: the game is over
                break
            human_move = chess.Move.from_uci(itos[token])
            yield game_id, half_move, board, human_move
            board.push(human_move)


def token_to_legal_move(token, board, itos):
    """Returns the move for this token, or None if it is not a legal move on this board."""
    if token < FIRST_MOVE_TOKEN:
        return None
    move = chess.Move.from_uci(itos[token])
    if move in board.legal_moves:
        return move
    return None


def bucket_start(half_move):
    """First half-move of the group this half-move belongs to, counting from 1: 0-19 -> 1, 20-39 -> 21."""
    return half_move // BUCKET_SIZE * BUCKET_SIZE + 1
