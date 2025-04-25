import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

#MAX_COMPOSERS = 48

# ─── 1) UTILS: load all lines from a split directory ───────────────────────────
def load_split_lines(split_dir: Path):
    """
    Returns:
      - all_lines: [str, …] the event tokens
      - comp_ids:  [int, …] the composer index for each line
      - composer2id: dict[str, int]
    """
    files = sorted(split_dir.glob("*.txt"))  # consistent ordering
    all_lines, comp_ids = [], []
    composer2id = {}

    for composer_idx, f in enumerate(files):
        composer2id[f.stem] = composer_idx
        lines = [l.strip() for l in open(f, 'r') if l.strip()]
        all_lines.extend(lines)
        comp_ids.extend([composer_idx] * len(lines))

    print(f"Num of composers detected: {len(composer2id)}")
    return all_lines, comp_ids, composer2id

# ─── 2) VOCAB BUILDER ─────────────────────────────────────────────────────────
def build_vocab(all_tokens):
    uniq = sorted(set(all_tokens))
    tok2idx = {tok: i for i, tok in enumerate(uniq)}
    idx2tok = {i: tok for tok, i in tok2idx.items()}
    return tok2idx, idx2tok

# ─── 3) DATASET ───────────────────────────────────────────────────────────────
class EventDataset(Dataset):
    def __init__(self, token_ids, comp_ids, seq_len=128):
        self.ids = token_ids
        self.comp_ids = comp_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids) - self.seq_len

    def __getitem__(self, i):
        x = torch.tensor(self.ids[i : i + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1 : i + self.seq_len + 1], dtype=torch.long)
        c = torch.tensor(self.comp_ids[i + self.seq_len], dtype=torch.long)
        return x, y, c

# ─── 4) MODEL ─────────────────────────────────────────────────────────────────
class EventLSTM(nn.Module):
    def __init__(self, vocab_size, n_composers, emb_dim=128, comp_emb_dim=32, hidden=256, layers=2):
        super().__init__()
        self.note_emb = nn.Embedding(vocab_size, emb_dim)
        self.comp_emb = nn.Embedding(n_composers, comp_emb_dim)
        self.lstm = nn.LSTM(
            emb_dim + comp_emb_dim, hidden, layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, comp_id, hidden=None):
        # x: [B,T], comp_id: [B]
        B, T = x.size()
        x = self.note_emb(x)                        # [B,T,emb_dim]
        c = self.comp_emb(comp_id).unsqueeze(1)     # [B,1,comp_emb_dim]
        c = c.expand(-1, T, -1)                    # [B,T,comp_emb_dim]
        x = torch.cat([x, c], dim=-1)               # [B,T,emb+comp]
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden               # [B,T,vocab]

# ─── 5) TRAIN / EVAL LOOP ─────────────────────────────────────────────────────
def run_epoch(model, dataloader, loss_fn, opt=None, device="cpu"):
    training = opt is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Train" if training else "Eval ", leave=False)
    for x, y, comp_id in loop:
        x, y, comp_id = x.to(device), y.to(device), comp_id.to(device)
        logits, _ = model(x, comp_id)
        B, T, V = logits.size()
        loss = loss_fn(logits.view(B*T, V), y.view(B*T))
        if training:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

# ─── 6) MAIN ──────────────────────────────────────────────────────────────────
def main():
    base = Path("event_sequences")
    train_lines, train_comps, composer2id = load_split_lines(base / "train")
    val_lines,   val_comps, _   = load_split_lines(base / "validation")
    test_lines,  test_comps, _  = load_split_lines(base / "test")

    tok2idx, idx2tok = build_vocab(train_lines + val_lines)
    print(f"Vocab size: {len(tok2idx)} tokens")
    
    save_dir = Path("event_sequences")
    save_dir.mkdir(exist_ok=True)

    torch.save(tok2idx, save_dir / "tok2idx.pt")
    torch.save(idx2tok, save_dir / "idx2tok.pt")
    torch.save(composer2id, save_dir / "composer2id.pt")
    torch.save({v: k for k, v in composer2id.items()}, save_dir / "id2composer.pt")

    train_ids = [tok2idx[t] for t in train_lines if t in tok2idx]
    val_ids   = [tok2idx[t] for t in val_lines   if t in tok2idx]
    test_ids  = [tok2idx[t] for t in test_lines  if t in tok2idx]
    train_comps = [c for t,c in zip(train_lines, train_comps) if t in tok2idx]
    val_comps   = [c for t,c in zip(val_lines,   val_comps)   if t in tok2idx]
    test_comps  = [c for t,c in zip(test_lines,  test_comps)  if t in tok2idx]

    seq_len = 128; batch = 64
    train_ds = EventDataset(train_ids, train_comps, seq_len)
    val_ds   = EventDataset(val_ids,   val_comps,   seq_len)
    test_ds  = EventDataset(test_ids,  test_comps,  seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EventLSTM(len(tok2idx), len(composer2id)).to(device)
    loss_fn= nn.CrossEntropyLoss()
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    best_val = float('inf')
    for epoch in range(1, 6):
        print(f"== Epoch {epoch} ==")
        train_loss = run_epoch(model, train_loader, loss_fn, opt, device)
        val_loss   = run_epoch(model, val_loader,   loss_fn, None, device)
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "event_sequences/best_event_lstm.pt")
            print("Saved new best model.")

    # Save loss plot
    plt.figure()
    plt.plot(range(1, 6), train_losses, label="Train Loss")
    plt.plot(range(1, 6), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("event_sequences/loss_plot.png")
    plt.close()

    # final test
    model.load_state_dict(torch.load("event_sequences/best_event_lstm.pt", weights_only=True))
    test_loss = run_epoch(model, test_loader, loss_fn, None, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main()
