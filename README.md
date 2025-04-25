# EC523_Project
# Composer-Conditioned Music Generation

This project explores **composer-conditioned symbolic music generation** using several deep generative models:

- **LSTM-based models** for sequential music prediction
- **RNNs** for event-based sequence modeling
- **Variational Autoencoders (VAEs)** for latent-space interpolation and creative music variation
- **Generative Adversarial Networks (GANs)** for adversarial music synthesis

Given a sequence of musical events and a **composer identity** (e.g., Bach, Mozart, Chopin), the models learn to generate stylistically consistent music.

Composer conditioning is implemented through dedicated **embedding layers**, enabling style transfer and stylistic control at generation time.

---

## Features

- **Flexible event-based music representation** (notes, durations, velocities — not raw audio)
- **Automatic composer detection**: dynamically handles any number of composers
- **Composer-conditioned generation**: easily switch between composers when generating music
- **Training and generation scripts** included
- **MIDI output**: converts generated event sequences into `.mid` files
- **Loss tracking and model checkpointing** built-in
- **Support for multiple model types**: LSTM, RNN, VAE, and GAN

---

## Models Supported

- `LSTM`
- `RNN`
- `VAE`
- `GAN`

---

## How it Works

1. **Prepare your dataset**  
   Structure your dataset with one `.txt` file per composer. Each file should contain symbolic event sequences (one event per line).

2. **Train a model**  
   Select your desired model (e.g., LSTM, VAE, or GAN) and train it with composer conditioning.

3. **Generate new music**  
   Provide a seed sequence and a composer ID to generate stylistically matched music.

4. **Listen to the result**  
   Generated `.txt` event sequences are automatically converted into `.mid` files for easy playback.

---

## Example Commands

**Training a model:**
```bash
python train.py --model lstm --epochs 10 --batch_size 64

**Generating music for a specific composer:**
```bash
python generate.py --composer_id 3 --steps 500 --temp 1.0
**Generating music for all composers:**
python generate.py --gen_all


