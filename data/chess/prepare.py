"""
prepare.py - Lichess (Hugging Face) -> nanoGPT-Daten (Move-Level UCI, ein Spiel pro festem Slot)

Jedes Spiel wird so abgelegt (aufgefuellt auf BLOCK_SIZE + 1 Tokens):

    <start> e2e4 e7e5 g1f3 ... <end> <pad> <pad> ...

Zu lange Spiele werden abgeschnitten und fuellen den Slot ohne <end> ganz aus.

Verworfen werden Bot-Partien, unfertige Partien, andere Varianten, Spiele mit
Elo ausserhalb des Bereichs, zu wenigen Zuegen, illegalen Zuegen und Duplikate.

Ausgabe: train.bin, val.bin und test.bin (nur fuer die Evaluation, train.py nutzt sie nicht).

Aufruf (zuerst Pro, dann Normal mit den train-Zuegen aus der Pro-Zusammenfassung):
    python prepare.py --out-dir data/chess_pro --dataset Lichess/tournament-chess-games --min-elo 2400
    python prepare.py --out-dir data/chess_normal --min-elo 1200 --max-elo 1500 --max-train-moves 8000000 \
        --data-files "data/year=2025/month=09/*.parquet"

Abhaengigkeiten:  pip install chess numpy datasets
"""
import argparse
import os
import pickle
import random
import re
import sys

import chess
import numpy as np
from datasets import load_dataset

# ------------------------------------------------------------ Einstellungen
DATASET = "Lichess/standard-chess-games"
MIN_ELO = 1400       # Spiele, in denen ein Spieler darunter liegt, werden verworfen
BLOCK_SIZE = 128     # Kontextlaenge = Cap fuer die Spiellaenge
MIN_MOVES = 10       # Spiele mit weniger Halbzuegen werden verworfen
VAL_FRAC = 0.005     # Anteil der Spiele fuer die Validierung
TEST_FRAC = 0.02     # Anteil der Spiele fuer die Evaluation
SHUFFLE_BUFFER = 10_000   # damit --max-train-moves nicht nur die ersten Partien nimmt

PAD, START, END = 0, 1, 2


# ------------------------------------------------------------ Vokabular
def build_vocab():
    """<pad>, <start>, <end> + alle 1968 moeglichen UCI-Zuege."""
    moves = set()
    for a in range(64):
        for b in range(64):
            df = abs(chess.square_file(a) - chess.square_file(b))
            dr = abs(chess.square_rank(a) - chess.square_rank(b))
            if a != b and (df == 0 or dr == 0 or df == dr or (df, dr) in ((1, 2), (2, 1))):
                moves.add(chess.Move(a, b).uci())
    for rank_from, rank_to in ((6, 7), (1, 0)):          # Umwandlungen (Weiss / Schwarz)
        for f in range(8):
            for to_f in (f - 1, f, f + 1):
                if 0 <= to_f < 8:
                    for piece in "qrbn":
                        moves.add(chess.square_name(chess.square(f, rank_from))
                                  + chess.square_name(chess.square(to_f, rank_to)) + piece)
    tokens = ["<pad>", "<start>", "<end>"] + sorted(moves)
    return {t: i for i, t in enumerate(tokens)}, dict(enumerate(tokens))


# ------------------------------------------------------------ Partien lesen
def read_games(dataset, data_files=None):
    """Streamt Partien von Hugging Face, ohne den Datensatz herunterzuladen."""
    ds = load_dataset(dataset, data_files=data_files, split="train", streaming=True)
    return ds.shuffle(seed=1337, buffer_size=SHUFFLE_BUFFER)


def is_playable(game):
    """Nur fertige Standard-Partien zwischen Menschen."""
    return ("BOT" not in (game["WhiteTitle"], game["BlackTitle"])
            and game["Result"] != "*"
            and game.get("Variant") in (None, "Standard"))        # nur im Turnier-Datensatz


def san_moves(movetext):
    """'1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 1-0' -> ['e4', 'e5', 'Nf3']"""
    text = re.sub(r"\{[^}]*\}", " ", movetext)       # Kommentare entfernen
    moves = []
    for tok in text.split():
        tok = re.sub(r"^\d+\.+", "", tok)             # Zugnummern ("12." / "12...")
        tok = tok.rstrip("?!")                        # Bewertungen
        if tok and not tok.startswith("$") and tok not in ("1-0", "0-1", "1/2-1/2", "*"):
            moves.append(tok)
    return moves


def elo_in_range(game, min_elo, max_elo):
    """Beide Spieler muessen im Bereich liegen; fehlende Elo zaehlt als 0."""
    elos = [game["WhiteElo"] or 0, game["BlackElo"] or 0]
    return min(elos) >= min_elo and (max_elo is None or max(elos) <= max_elo)


# ------------------------------------------------------------ Hauptprogramm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET, help="Hugging-Face-Datensatz")
    ap.add_argument("--data-files", help='z. B. "data/year=2025/month=09/*.parquet"')
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--min-elo", type=int, default=MIN_ELO)
    ap.add_argument("--max-elo", type=int)
    ap.add_argument("--max-train-moves", type=int, help="Stoppt, sobald train so viele Zuege enthaelt")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = ap.parse_args()

    stoi, itos = build_vocab()
    slot_len = args.block_size + 1
    max_moves = slot_len - 2                          # Platz fuer <start> und <end>
    os.makedirs(args.out_dir, exist_ok=True)
    files = {s: open(os.path.join(args.out_dir, f"{s}.bin"), "wb") for s in ("train", "val", "test")}

    rng = random.Random(1337)
    known_games = set()
    n = dict(seen=0, not_playable=0, elo=0, short=0, illegal=0, duplicate=0, truncated=0,
             train=0, val=0, test=0, train_moves=0)

    print(f"Lese {args.dataset} {args.data_files or ''} ...")
    for game in read_games(args.dataset, args.data_files):
        n["seen"] += 1
        if n["seen"] % 100_000 == 0:
            print(f"  {n['seen']:,} gelesen, {n['train'] + n['val'] + n['test']:,} behalten")

        if not is_playable(game):
            n["not_playable"] += 1
            continue
        if not elo_in_range(game, args.min_elo, args.max_elo):
            n["elo"] += 1
            continue
        moves = san_moves(game["movetext"] or "")
        if len(moves) < MIN_MOVES:
            n["short"] += 1
            continue
        truncated = len(moves) > max_moves
        if truncated:                                 # voller Slot ohne <end>
            moves = moves[:slot_len - 1]

        # Spiel auf einem Brett abspielen: SAN -> UCI, dabei Legalitaet pruefen
        board = chess.Board()
        ids = [START]
        try:
            for san in moves:
                move = board.parse_san(san)
                ids.append(stoi[move.uci()])
                board.push(move)
        except (ValueError, KeyError):                # KeyError: Nullzug ("--") nicht im Vokabular
            n["illegal"] += 1
            continue
        if not truncated:
            ids.append(END)

        slot = np.full(slot_len, PAD, dtype=np.uint16)
        slot[:len(ids)] = ids

        # Duplikate vor dem Aufteilen entfernen, damit kein Spiel in zwei Splits landet
        key = hash(slot.tobytes())
        if key in known_games:
            n["duplicate"] += 1
            continue
        known_games.add(key)
        n["truncated"] += truncated

        r = rng.random()
        split = "test" if r < TEST_FRAC else "val" if r < TEST_FRAC + VAL_FRAC else "train"
        files[split].write(slot.tobytes())
        n[split] += 1
        if split == "train":
            n["train_moves"] += len(moves)
            if args.max_train_moves and n["train_moves"] >= args.max_train_moves:
                break

    for f in files.values():
        f.close()
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(stoi), "stoi": stoi, "itos": itos,
                     "block_size": args.block_size}, f)

    print("\nFertig.")
    print(f"  gelesen:                  {n['seen']:,}")
    print(f"  verworfen (Bot/unfertig/Variante): {n['not_playable']:,}")
    print(f"  verworfen (Elo):          {n['elo']:,}")
    print(f"  verworfen (zu kurz):      {n['short']:,}")
    print(f"  verworfen (illegal):      {n['illegal']:,}")
    print(f"  verworfen (Duplikate):    {n['duplicate']:,}")
    print(f"  abgeschnitten (> {max_moves} Halbzuege): {n['truncated']:,}")
    print(f"  train: {n['train']:,}   val: {n['val']:,}   test: {n['test']:,}   vocab_size: {len(stoi)}")
    print(f"  train-Zuege: {n['train_moves']:,}")

    # Die Dateien sind geschlossen. Das Streaming kann beim normalen Beenden abstuerzen.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()