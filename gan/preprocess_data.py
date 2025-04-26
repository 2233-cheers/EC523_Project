import os

import pretty_midi
import numpy as np


def midi_to_pianoroll(midi_file, time_steps=128, fs=32):
    """
    Convert MIDI file to piano roll format
    :param midi_file: MIDI file path
    :param time_steps: output time steps
    :param fs: sampling frequency per second (time resolution)
    :return: pianoroll (time_steps, pitch_range)
    """
    # Loading MIDI Files
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Get piano note range (MIDI 21-108, corresponding to 88 keys)
    pitch_range = 88
    pianoroll = np.zeros((time_steps, pitch_range))

    # Calculate the length of each time step
    total_duration = midi_data.get_end_time()  # Total MIDI duration (seconds)
    time_per_step = total_duration / time_steps

    # Iterate over all notes
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                # Mapping note times to time steps
                start_step = int(note.start / time_per_step)
                end_step = int(note.end / time_per_step)
                pitch_idx = note.pitch - 21  # MIDI 21 corresponds to index 0

                # Make sure the index is in range
                if pitch_idx < 0 or pitch_idx >= pitch_range:
                    continue
                start_step = min(start_step, time_steps - 1)
                end_step = min(end_step, time_steps)

                # Marking notes in the piano roll
                pianoroll[start_step:end_step, pitch_idx] = 1

    return pianoroll


input_dir = "organized_midi"  # Raw MIDI folders by composer
output_dir = "processed_rolls"
os.makedirs(output_dir, exist_ok=True)

for composer in os.listdir(input_dir):
    composer_dir = os.path.join(input_dir, composer)
    if not os.path.isdir(composer_dir):
        continue

    for fname in os.listdir(composer_dir):
        if not fname.endswith(".mid") and not fname.endswith(".midi"):
            continue

        midi_path = os.path.join(composer_dir, fname)
        try:
            roll = midi_to_pianoroll(midi_path, time_steps=96, fs=32)
            if roll.shape != (96, 88):
                continue  # 跳过短片段
            save_name = f"{composer}_{os.path.splitext(fname)[0]}.npy"
            np.save(os.path.join(output_dir, save_name), roll)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
