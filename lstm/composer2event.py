import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from midi2event import midi_to_event_sequence

def main(root_dir: str, csv_path: str):
    root = Path(root_dir).resolve()
    csv_df = pd.read_csv(csv_path)
    csv_df["base"] = csv_df["midi_filename"].apply(lambda x: Path(x).name)
    fn2split = dict(zip(csv_df["base"], csv_df["split"]))

    # collect per‑composer, per‑split event sequences
    composer_splits: dict[str, dict[str, list[list[str]]]] = {}

    for midi_path in tqdm(root.glob("**/*.mid*"), desc="Scanning MIDI"):
        base = midi_path.name
        split = fn2split.get(base)
        if split not in ("train", "validation", "test"):
            continue

        composer = midi_path.parent.name
        events = midi_to_event_sequence(str(midi_path))

        composer_splits.setdefault(composer, {"train": [], "validation": [], "test": []})
        composer_splits[composer][split].append(events)

    # base output directory
    out_root = Path("event_sequences")
    out_root.mkdir(exist_ok=True, parents=True)

    # create subfolders and write out per‑composer files
    for split in ("train", "validation", "test"):
        split_dir = out_root / split
        split_dir.mkdir(exist_ok=True, parents=True)
        for composer, splits in composer_splits.items():
            events_lists = splits[split]
            if not events_lists:
                continue
            out_file = split_dir / f"{composer}.txt"
            with open(out_file, "w") as fw:
                for events in events_lists:
                    fw.write("BAR\n")  # separator between pieces
                    for ev in events:
                        fw.write(ev + "\n")
            print(f"Wrote {len(events_lists)} {split} pieces for {composer} → {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate per‑composer train/validation/test event TXT from MIDI library"
    )
    p.add_argument("--root_dir", required=True,
                   help="Root dir containing Composer/*composer/*.mid*")
    p.add_argument("--csv", required=True,
                   help="CSV with columns: midi_filename,split ('train','validation','test')")
    args = p.parse_args()
    main(args.root_dir, args.csv)
