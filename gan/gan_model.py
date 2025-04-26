import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_shape=(96, 88), num_classes=30):
        super().__init__()
        self.out_shape = out_shape
        self.latent_dim = latent_dim
        self.label_embed = nn.Embedding(num_classes, latent_dim)

        total_dim = latent_dim + latent_dim  # z + label embedding
        self.model = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_shape[0] * out_shape[1]),
            nn.Sigmoid()  # piano-roll → [0,1]
        )

    def forward(self, z, labels):
        label_embed = self.label_embed(labels)
        x = torch.cat((z, label_embed), dim=1)
        out = self.model(x)
        return out.view(-1, *self.out_shape)  # [B, T, 88]


class Discriminator(nn.Module):
    def __init__(self, input_shape=(96, 88), num_classes=30):
        super().__init__()
        self.input_dim = input_shape[0] * input_shape[1]
        self.label_embed = nn.Embedding(num_classes, self.input_dim)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)  # flatten
        cond = self.label_embed(labels)
        x = x + cond
        return self.model(x)
