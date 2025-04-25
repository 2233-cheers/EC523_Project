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
**Saving dataset to event keys:**
```bash
python composer2event.py --csv ../../data/maestro-v3.0.0.csv --root_dir ../../data/Composer/
```
**Training a model:**
```bash
python model.py
```
**Generating music for a specific composer:**
```bash
python generate.py --composer_id 3 --steps 500 --temp 1.0
```
**Generating music for all composers:**
```bash
python generate.py --gen_all --steps 2500 --temp 1 --seed_file start.txt
```

---

## Data Format

Each music piece is represented as a **sequence of symbolic events**, not raw audio.  
This allows flexible and lightweight modeling of musical structure.

Example snippet of the event format:
BAR QPM 120.00 VELOCITY 21 NOTE_ON 59 TIME_SHIFT 0.006 VELOCITY 61 NOTE_ON 71 TIME_SHIFT 0.010 VELOCITY 36 NOTE_ON 43 ...

**Explanation of the events:**

- `BAR` — Marks the start of a new measure (musical bar).
- `QPM <float>` — "Quarter notes per minute" (i.e., tempo).
- `VELOCITY <int>` — The volume (strength) of the next note(s).
- `NOTE_ON <int>` — Start playing a note with a given MIDI pitch number (e.g., 59 = B3).
- `NOTE_OFF <int>` — Stop playing a note with a given MIDI pitch number.
- `TIME_SHIFT <float>` — Wait for a certain amount of seconds before the next event.

**Key Points:**
- Sequences can have variable length.
- Timing and note dynamics are captured explicitly.
- This format is easily convertible into standard `.mid` MIDI files for playback.


