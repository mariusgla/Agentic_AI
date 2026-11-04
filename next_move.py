# next_move.py  (im NanoGPT-Ordner)
import sys, torch, chess
from model import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD, START, END = 0, 1, 2

def build_vocab():
    """<pad>, <start>, <end> + alle 1968 moeglichen UCI-Zuege."""
    moves = set()
    for a in range(64):
        for b in range(64):
            df = abs(chess.square_file(a) - chess.square_file(b))
            dr = abs(chess.square_rank(a) - chess.square_rank(b))
            if a != b and (df == 0 or dr == 0 or df == dr or (df, dr) in ((1, 2), (2, 1))):
                moves.add(chess.Move(a, b).uci())
    for rank_from, rank_to in ((6, 7), (1, 0)):
        for f in range(8):
            for to_f in (f - 1, f, f + 1):
                if 0 <= to_f < 8:
                    for piece in "qrbn":
                        moves.add(chess.square_name(chess.square(f, rank_from))
                                  + chess.square_name(chess.square(to_f, rank_to)) + piece)
    tokens = ["<pad>", "<start>", "<end>"] + sorted(moves)
    return {t: i for i, t in enumerate(tokens)}, dict(enumerate(tokens))

move2id, id2move = build_vocab()

# Modell laden
ckpt = torch.load('out-chess/ckpt.pt', map_location=device)
model = GPT(GPTConfig(**ckpt['model_args']))
model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()})
model.eval().to(device)

@torch.no_grad()
def predict(moves, k=5):
    idx = torch.tensor([[START] + [move2id[m] for m in moves]], device=device)
    logits, _ = model(idx)
    probs = torch.softmax(logits[0, -1], -1)
    top = torch.topk(probs, k)
    return [(id2move[i.item()], p.item()) for p, i in zip(top.values, top.indices)]

moves = sys.argv[1:] or input("Züge (UCI, durch Leerzeichen getrennt): ").split()
for m, p in predict(moves):
    print(f"{m:8s} {p:6.1%}")