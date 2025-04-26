import re
import matplotlib.pyplot as plt

log_text = 'train_log.txt'

# Use regular expressions to parse Epoch, D_loss, G_loss from logs
pattern = r"\[Epoch\s+(\d+)\]\s+D_loss:\s+([\d\.]+)\s+\|\s+G_loss:\s+([\d\.]+)\s+\|\s+CosSim:\s+([\d\.]+)"

epochs = []
d_losses = []
g_losses = []
cos_sims = []

# 读取并解析文件
with open(log_text, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            d_losses.append(float(match.group(2)))
            g_losses.append(float(match.group(3)))
            cos_sims.append(float(match.group(4)))

# 画图 - D_loss 和 G_loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, d_losses, label='Discriminator Loss')
plt.plot(epochs, g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epochs, cos_sims, label='Cosine Similarity')
plt.xlabel('Epoch')
plt.ylabel('CosSim')
plt.title('Cosine Similarity over Epochs')
plt.legend()
plt.show()

