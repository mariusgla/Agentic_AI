"""
Main eval: predicts once, then scores legal move rate and Stockfish move quality.

Run from the repo root:
    python eval/run_eval.py --ckpt out-chess/test-pro/ckpt.pt --data data/chess_pro

Saves a CSV and a PNG per metric next to the checkpoint, named after the dataset and split.
"""
import argparse
import os

import common
import legal_moves
import stockfish_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-games", type=int, help="Use only the first games, for quick tests")
    parser.add_argument("--no-stockfish", action="store_true", help="Only score the legal move rate")
    return parser.parse_args()


def run_legal_moves(games, predictions, itos, output_base, title):
    positions = legal_moves.score(games, predictions, itos)
    summary = legal_moves.summarize(positions)

    print(f"{len(games)} games, {len(positions)} positions")
    print(summary.round(1).to_string(index=False))
    print(f"overall: {100 * positions['legal'].mean():.1f}% legal\n")

    summary.to_csv(f"{output_base}.csv", index=False)
    legal_moves.plot(summary, title, f"{output_base}.png")
    print(f"saved {output_base}.csv and {output_base}.png\n")


def run_stockfish(games, predictions, itos, output_base, title):
    results = stockfish_eval.score(games, predictions, itos)
    summary = stockfish_eval.summarize(results)

    print(summary.to_string(index=False))

    results.to_csv(f"{output_base}.csv", index=False)
    stockfish_eval.plot(summary, title, f"{output_base}.png")
    print(f"saved {output_base}.csv and {output_base}.png")


def main():
    args = parse_args()

    games, itos = common.load_games(args.data, args.split)
    games = games[:args.max_games]
    model = common.load_model(args.ckpt)
    predictions = common.predict(model, games)

    dataset = os.path.basename(os.path.normpath(args.data))
    out_dir = os.path.dirname(args.ckpt)
    described = f"on {dataset} {args.split} ({len(games)} games)"

    run_legal_moves(games, predictions, itos,
                    output_base=os.path.join(out_dir, f"legal_moves_{dataset}_{args.split}"),
                    title=f"Legal move rate {described}")

    if not args.no_stockfish:
        run_stockfish(games, predictions, itos,
                      output_base=os.path.join(out_dir, f"stockfish_{dataset}_{args.split}"),
                      title=f"Move quality {described}")


if __name__ == "__main__":
    main()
