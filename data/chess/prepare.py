"""
prepare.py - Lichess PGN -> nanoGPT-Daten (Move-Level UCI, ein Spiel pro festem Slot)

Jedes Spiel wird so abgelegt (aufgefuellt auf BLOCK_SIZE + 1 Tokens):

    <start> e2e4 e7e5 g1f3 ... <end> <pad> <pad> ...

Verworfen werden Spiele mit zu niedriger Elo, zu wenigen/zu vielen Zuegen
und Spiele mit illegalen Zuegen.

Aufruf:
    python prepare.py lichess_2014-06.pgn.zst
    python prepare.py a.pgn.zst b.pgn.zst --min-elo 1800 --block-size 256 --out-dir data/chess

Abhaengigkeiten:  pip install chess numpy zstandard
"""
import argparse
import io
import os
import pickle
import random
import re

import chess
import numpy as np

# ------------------------------------------------------------ Einstellungen
MIN_ELO = 1400       # Spiele, in denen ein Spieler darunter liegt, werden verworfen
BLOCK_SIZE = 128     # Kontextlaenge = Cap fuer die Spiellaenge
MIN_MOVES = 10       # Spiele mit weniger Halbzuegen werden verworfen
VAL_FRAC = 0.005     # Anteil der Spiele fuer die Validierung

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


# ------------------------------------------------------------ PGN lesen
def open_text(path):
    if path.endswith(".zst"):
        import zstandard
        reader = zstandard.ZstdDecompressor().stream_reader(open(path, "rb"))
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def read_games(f):
    """Liefert (Header-Dict, Zugzeile) pro Partie."""
    headers = {}
    for line in f:
        line = line.strip()
        if line.startswith("["):
            m = re.match(r'\[(\w+) "(.*)"\]', line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line:
            yield headers, line
            headers = {}


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


def min_elo_of(headers):
    elos = [headers.get("WhiteElo", ""), headers.get("BlackElo", "")]
    return min(int(e) if e.isdigit() else 0 for e in elos)


# ------------------------------------------------------------ Hauptprogramm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help=".pgn oder .pgn.zst")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--min-elo", type=int, default=MIN_ELO)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = ap.parse_args()

    stoi, itos = build_vocab()
    slot_len = args.block_size + 1
    max_moves = slot_len - 2                          # Platz fuer <start> und <end>
    os.makedirs(args.out_dir, exist_ok=True)
    train_f = open(os.path.join(args.out_dir, "train.bin"), "wb")
    val_f = open(os.path.join(args.out_dir, "val.bin"), "wb")

    rng = random.Random(1337)
    n = dict(seen=0, low_elo=0, short=0, long=0, illegal=0, train=0, val=0)

    for path in args.inputs:
        print(f"Lese {path} ...")
        with open_text(path) as f:
            for headers, movetext in read_games(f):
                n["seen"] += 1
                if n["seen"] % 100_000 == 0:
                    print(f"  {n['seen']:,} gelesen, {n['train'] + n['val']:,} behalten")

                if min_elo_of(headers) < args.min_elo:
                    n["low_elo"] += 1
                    continue
                moves = san_moves(movetext)
                if len(moves) < MIN_MOVES:
                    n["short"] += 1
                    continue
                if len(moves) > max_moves:
                    n["long"] += 1
                    continue

                # Spiel auf einem Brett abspielen: SAN -> UCI, dabei Legalitaet pruefen
                board = chess.Board()
                ids = [START]
                try:
                    for san in moves:
                        move = board.parse_san(san)
                        ids.append(stoi[move.uci()])
                        board.push(move)
                except ValueError:
                    n["illegal"] += 1
                    continue
                ids.append(END)

                slot = np.full(slot_len, PAD, dtype=np.uint16)
                slot[:len(ids)] = ids
                split = "val" if rng.random() < VAL_FRAC else "train"
                (val_f if split == "val" else train_f).write(slot.tobytes())
                n[split] += 1

    train_f.close()
    val_f.close()
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(stoi), "stoi": stoi, "itos": itos,
                     "block_size": args.block_size}, f)

    print("\nFertig.")
    print(f"  gelesen:                  {n['seen']:,}")
    print(f"  verworfen (Elo < {args.min_elo}):  {n['low_elo']:,}")
    print(f"  verworfen (zu kurz):      {n['short']:,}")
    print(f"  verworfen (> {max_moves} Halbzuege): {n['long']:,}")
    print(f"  verworfen (illegal):      {n['illegal']:,}")
    print(f"  train: {n['train']:,}   val: {n['val']:,}   vocab_size: {len(stoi)}")


if __name__ == "__main__":
    main()