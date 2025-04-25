import pretty_midi
from collections import namedtuple

Event = namedtuple("Event", ["time", "type", "pitch", "velocity"])

def midi_to_event_sequence(midi_path):
    """
    Read a MIDI file and return an event sequence as a list of strings:
      - QPM <tempo>
      - TIME_SHIFT <seconds>
      - VELOCITY <velocity>
      - NOTE_ON <pitch>
      - NOTE_OFF <pitch>
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Extract tempo (QPM)
    _, qpms = midi.get_tempo_changes()
    qpm = float(qpms[0]) if len(qpms) else 120.0

    # Collect on/off events with timing
    evs = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            evs.append(Event(note.start, "ON",  note.pitch, note.velocity))
            evs.append(Event(note.end,   "OFF", note.pitch, None))

    # Sort events by time
    evs.sort(key=lambda e: e.time)

    # Build event lines
    lines = [f"QPM {qpm:.2f}"]
    last_t = 0.0
    for e in evs:
        dt = e.time - last_t
        lines.append(f"TIME_SHIFT {dt:.3f}")
        if e.type == "ON":
            lines.append(f"VELOCITY {e.velocity}")
            lines.append(f"NOTE_ON {e.pitch}")
        else:
            lines.append(f"NOTE_OFF {e.pitch}")
        last_t = e.time

    return lines
