import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pretty_midi
from pathlib import Path

# === CONFIG ===
BASE_DIR    = r"C:\Users\zzz\Desktop\EC523\Project\MusicGen_Project"
MIDI_DIR    = os.path.join(BASE_DIR, "Composer", "Claude_Debussy")
VOCAB_DIR   = os.path.join(BASE_DIR, "event_vocab")
OUTPUT_DIR  = os.path.join(BASE_DIR, "Generated")

EMB_SIZE    = 128
SEQ_LEN     = 128
LATENT_DIM  = 64
BATCH_SIZE  = 64
EPOCHS      = 1
LR          = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def midi_to_event_sequence(midi_path, quantize_dt=0.01):
    midi = pretty_midi.PrettyMIDI(midi_path)
    _, qpms = midi.get_tempo_changes()
    qpm = float(qpms[0]) if len(qpms) else 120.0

    evs = []
    for inst in midi.instruments:
        if inst.is_drum: continue
        for note in inst.notes:
            evs.append((note.start,  "ON",  note.pitch, note.velocity))
            evs.append((note.end,    "OFF", note.pitch, None))

    evs.sort(key=lambda e: e[0])
    tokens = [f"QPM_{qpm:.2f}"]
    last_t = 0.0
    for e in evs:
        dt = e[0] - last_t
        dt_q = round(dt / quantize_dt) * quantize_dt
        tokens.append(f"TIME_SHIFT_{dt_q:.3f}")
        if e[1] == "ON":
            tokens.append(f"VELOCITY_{e[3]}")
            tokens.append(f"NOTE_ON_{e[2]}")
        else:
            tokens.append(f"NOTE_OFF_{e[2]}")
        last_t = e[0]
    return tokens

tok2idx = json.load(open(f"{VOCAB_DIR}/tok2idx.json"))
idx2tok = json.load(open(f"{VOCAB_DIR}/idx2tok.json"))
emb_matrix = np.load(f"{VOCAB_DIR}/embeddings.npy")
vocab_size = len(tok2idx)

class EventDataset(Dataset):
    def __init__(self, midi_dir):
        lines = []
        for fn in os.listdir(midi_dir):
            if fn.lower().endswith((".mid",".midi")):
                seq = midi_to_event_sequence(os.path.join(midi_dir, fn))
                seq_ids = [tok2idx[t] for t in seq if t in tok2idx]
                lines.extend(seq_ids)
        self.ids = lines

    def __len__(self):
        return len(self.ids) - SEQ_LEN

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+SEQ_LEN], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+SEQ_LEN+1], dtype=torch.long)
        return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2)*(-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:,:x.size(1)].to(x.device)

class EventVAE(nn.Module):
    def __init__(self, vocab_size, emb_size, latent_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight.data.copy_(torch.from_numpy(emb_matrix))
        self.emb.weight.requires_grad = True

        self.posenc = PositionalEncoding(emb_size)
        self.enc_fc1 = nn.Linear(SEQ_LEN*emb_size, 512)
        self.to_mu = nn.Linear(512, latent_dim)
        self.to_logvar = nn.Linear(512, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, SEQ_LEN*emb_size)
        self.out_fc = nn.Linear(emb_size, vocab_size)

    def encode(self, x):
        e = self.emb(x)
        e = self.posenc(e)
        e = e.view(e.size(0), -1)
        h = F.relu(self.enc_fc1(e))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        e = self.dec_fc2(h).view(-1, SEQ_LEN, EMB_SIZE)
        e = self.posenc(e)
        logits = self.out_fc(e)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

dataset = EventDataset(MIDI_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = EventVAE(vocab_size, EMB_SIZE, LATENT_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
bce = nn.CrossEntropyLoss()

for ep in range(EPOCHS):
    model.train()
    total = 0
    for x, y in tqdm(loader, desc=f"Epoch {ep+1}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, mu, logvar = model(x)
        B,T,V = logits.size()
        loss_rec = bce(logits.view(B*T, V), y.view(B*T))
        kl = -0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp()) / B
        loss = loss_rec + 1e-3 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {ep+1} Loss: {total/len(loader):.4f}")

Path(OUTPUT_DIR).mkdir(exist_ok=True)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "event_vae.pt"))
print("✔ 模型已保存到:", OUTPUT_DIR)

