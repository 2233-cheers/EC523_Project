import pretty_midi
from pathlib import Path

def event_sequence_to_midi(event_path,
                                  out_path="reconstructed.mid",
                                  default_qpm=120.0,
                                  velocity_default=80):
    # Load all non‐empty lines
    lines = [l.strip() for l in open(event_path) if l.strip()]

    # Read QPM from the first line if present
    idx = 0
    if lines[0].startswith("QPM"):
        qpm = float(lines[0].split()[1])
        idx = 1
    else:
        qpm = default_qpm

    # Prepare PrettyMIDI with that tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=qpm)
    inst = pretty_midi.Instrument(program=0)

    time_cursor     = 0.0
    active_notes    = []   # list of tuples (pitch, start_time, velocity)
    current_velocity = velocity_default

    # Process each event line
    while idx < len(lines):
        ev = lines[idx]
        idx += 1

        if ev.startswith("TIME_SHIFT"):
            # add real‐time delta (in seconds)
            delta = float(ev.split()[1])
            time_cursor += delta

        elif ev.startswith("VELOCITY"):
            current_velocity = int(ev.split()[1])

        elif ev.startswith("NOTE_ON"):
            pitch = int(ev.split()[1])
            # record note‑on at current time
            active_notes.append((pitch, time_cursor, current_velocity))

        elif ev.startswith("NOTE_OFF"):
            pitch = int(ev.split()[1])
            # match earliest active NOTE_ON for this pitch
            for i, (p, t0, vel) in enumerate(active_notes):
                if p == pitch:
                    # create and append the pretty_midi.Note
                    note = pretty_midi.Note(
                        velocity=vel,
                        pitch=p,
                        start=t0,
                        end=time_cursor
                    )
                    inst.notes.append(note)
                    active_notes.pop(i)
                    break
        # ignore other markers (BAR, POS, etc.)

    # Finalize and write
    midi.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(out_path)
    print(f"Reconstructed MIDI saved to: {out_path}")


if __name__ == "__main__":
    event_sequence_path = "embedding_outputs/midi_event_sequence.txt"
    event_sequence_to_midi(event_sequence_path)
