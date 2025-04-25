import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse

# ─── 1) MODEL: must match your training definition ─────────────────────────────────
from model import EventLSTM

# ─── 2) DECODER: unchanged ───────────────────────────────────────────────────────
from event2midi import event_sequence_to_midi

# ─── 3) GENERATION CORE ─────────────────────────────────────────────────────────

def generate_tokens(
    model: EventLSTM,
    idx2tok: dict[int, str],
    tok2idx: dict[str, int],
    seed_tokens: list[str],
    composer_id: int,
    total_steps: int,
    temperature: float,
    device: torch.device
) -> list[str]:
    model.eval()
    # encode seed
    seq = [tok2idx[t] for t in seed_tokens if t in tok2idx]
    if not seq:
        seq = [tok2idx.get("BAR", 0)]
    hidden = None

    for _ in range(total_steps):
        x = torch.tensor([seq[-1:]], dtype=torch.long, device=device)
        c = torch.tensor([composer_id], dtype=torch.long, device=device)
        logits, hidden = model(x, c, hidden)
        logit = logits[0,0] / temperature
        probs = F.softmax(logit, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        seq.append(nxt)

    return [idx2tok[i] for i in seq]

# ─── 4) CLI + MAIN ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate music with composer conditioning")
    p.add_argument("--model",       default="event_sequences/best_event_lstm.pt")
    p.add_argument("--tok2idx",     default="event_sequences/tok2idx.pt")
    p.add_argument("--idx2tok",     default="event_sequences/idx2tok.pt")
    p.add_argument("--id2composer", default="event_sequences/id2composer.pt")
    p.add_argument("--n_composers", type=int, default=4,
                   help="Number of composer embeddings the model was trained with")
    p.add_argument("--composer_id", type=int, default=0,
                   help="Which composer style to generate (0‑indexed)")
    p.add_argument("--gen_all",     action="store_true",
                   help="Generate one output per composer in [0..n_composers-1]")
    p.add_argument("--out_events",  default="generated_events.txt")
    p.add_argument("--out_midi",    default="generated.midi")
    p.add_argument("--steps",       type=int, default=500)
    p.add_argument("--temp",        type=float, default=1.0)
    p.add_argument("--seed_file",   default=None,
                   help="Path to file of seed events, one per line")
    p.add_argument("--seed",        default=None,
                   help="Inline seed (use '\\n' between lines) if no seed_file")
    args = p.parse_args()

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok2idx       = torch.load(args.tok2idx)
    idx2tok       = torch.load(args.idx2tok)
    id2composer   = torch.load(args.id2composer)  
    model         = EventLSTM(len(tok2idx), args.n_composers).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Seed
    if args.seed_file:
        seed_tokens = [l.strip() for l in open(args.seed_file) if l.strip()]
    elif args.seed:
        seed_tokens = [l for l in args.seed.split("\\n") if l]
    else:
        seed_tokens = ["BAR"]
    print("Seed:", seed_tokens)

    composers = range(args.n_composers) if args.gen_all else [args.composer_id]
    for cid in composers:
        composer_name = id2composer.get(cid, f"comp{cid}")
        print(f"Generating for composer_id={cid} ({composer_name})")

        events = generate_tokens(
            model, idx2tok, tok2idx,
            seed_tokens, cid,
            total_steps=args.steps,
            temperature=args.temp,
            device=device
        )

        # output file names
        out_e = Path(args.out_events)
        out_m = Path(args.out_midi)
        if args.gen_all:
            out_e = out_e.parent / f"{out_e.stem}_{composer_name}.txt"
            out_m = out_m.parent / f"{out_m.stem}_{composer_name}{out_m.suffix}"

        out_e.parent.mkdir(exist_ok=True, parents=True)
        out_e.write_text("\n".join(events))
        print("Events →", out_e)

        event_sequence_to_midi(str(out_e), out_path=str(out_m))
        print("MIDI  →", out_m)
