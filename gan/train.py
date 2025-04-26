import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PianoRollDataset
from gan_model import Generator, Discriminator
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# hyper-parameters
latent_dim = 100
roll_shape = (96, 88)
batch_size = 32
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
dataset = PianoRollDataset("processed_rolls")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.composer2id)
# initialize model
G = Generator(latent_dim, out_shape=roll_shape, num_classes=num_classes).to(device)
D = Discriminator(input_shape=roll_shape, num_classes=num_classes).to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

log_file = open("train_log.txt", "w")

# train
for epoch in range(epochs):
    g_total = 0
    d_total = 0

    for real, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        real = real + 0.05 * torch.rand_like(real)  # 添加少量噪声，避免 D 过强
        real = torch.clamp(real, 0.0, 1.0)

        real, labels = real.to(device), labels.to(device)
        valid = torch.ones((real.size(0), 1), device=device)
        fake = torch.zeros((real.size(0), 1), device=device)

        # Generator
        opt_G.zero_grad()
        z = torch.randn(real.size(0), latent_dim, device=device)
        gen = G(z, labels)
        g_loss = criterion(D(gen, labels), valid)
        g_loss.backward()
        opt_G.step()

        # Discriminator
        opt_D.zero_grad()
        real_loss = criterion(D(real, labels), valid)
        fake_loss = criterion(D(gen.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        g_total += g_loss.item()
        d_total += d_loss.item()

    # Cosine Similarity
    with torch.no_grad():
        z_sample = torch.randn(1, latent_dim, device=device)
        l_sample = torch.randint(0, num_classes, (1,), device=device)
        gen_sample = G(z_sample, l_sample)
        real_sample = real[:1]

        sim = cosine_similarity(gen_sample.view(1, -1).cpu(), real_sample.view(1, -1).cpu())[0][0]

    log_line = f"[Epoch {epoch + 1}] D_loss: {d_total:.4f} | G_loss: {g_total:.4f} | CosSim: {sim:.4f}"
    print(log_line)
    log_file.write(log_line + "\n")
    log_file.flush()

    if epoch + 1 == epochs:
        torch.save(G.state_dict(), "pianoroll_generator_final.pt")

log_file.close()


