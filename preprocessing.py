import pretty_midi
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import sys
import re
import yaml

# --- 1. Configuration (Relative) ---
ROOT_PATH = Path("dataset_root")
DOCS_PATH = ROOT_PATH / "docs"
PROCESSED_PATH = ROOT_PATH / "processed"

MAX_NOTES = 512
INSTRUMENT_PROGRAM = 0
PAD_VALUE = 0.0 # Use 0.0 for padding normalized data

MANIFEST_FILE = DOCS_PATH / 'data_manifest.csv'
PROCESSED_PATH.mkdir(exist_ok=True)

# --- NORMALIZATION CONSTANTS ---
# We will map raw MIDI values to a [-1, 1] range
# Pitch: 0-127 -> (val / 63.5) - 1.0
# Velocity: 0-127 -> (val / 63.5) - 1.0
# Duration/Step: 0-4 beats -> (val / 2.0) - 1.0 (clipping > 4)
MAX_BEAT_TIME = 4.0

# --- 2. Helper Functions ---
def sanitize_filename(stem):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)

def extract_features(midi_path, max_notes_const):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return None

    # --- Numeric Features (Un-normalized) ---
    tempo = midi_data.get_tempo_changes()[1][0] if midi_data.get_tempo_changes()[1].size > 0 else 120
    key_analysis = midi_data.key_signature_changes
    key_num = key_analysis[0].key_number if len(key_analysis) > 0 else 0

    all_notes = []
    for instrument in midi_data.instruments:
        if instrument.program == INSTRUMENT_PROGRAM:
            for note in instrument.notes:
                start_beat = midi_data.time_to_tick(note.start) / midi_data.resolution
                end_beat = midi_data.time_to_tick(note.end) / midi_data.resolution
                duration_beat = end_beat - start_beat

                # --- NOTE DATA NORMALIZATION ---
                norm_pitch = (note.pitch / 63.5) - 1.0
                norm_velocity = (note.velocity / 63.5) - 1.0

                # Clip duration at 4 beats, then normalize
                clipped_duration = min(duration_beat, MAX_BEAT_TIME)
                norm_duration = (clipped_duration / (MAX_BEAT_TIME / 2.0)) - 1.0

                all_notes.append([
                    norm_pitch, start_beat, norm_duration, norm_velocity
                ])

    if not all_notes:
        return None

    # Sort by start time to calculate step (delta)
    all_notes.sort(key=lambda x: x[1])

    note_array_final = []
    last_start_beat = 0.0
    for note_data in all_notes:
        norm_pitch, start_beat, norm_duration, norm_velocity = note_data

        # Calculate step time
        step_beat = start_beat - last_start_beat
        last_start_beat = start_beat

        # Clip step time at 4 beats, then normalize
        clipped_step = min(step_beat, MAX_BEAT_TIME)
        norm_step = (clipped_step / (MAX_BEAT_TIME / 2.0)) - 1.0

        # Final array structure: [Pitch, Velocity, Duration, Step]
        note_array_final.append([norm_pitch, norm_velocity, norm_duration, norm_step])

    note_array = np.array(note_array_final, dtype=np.float32)

    # --- Numeric Features (Mean of RAW data) ---
    raw_pitches = [n[0] for n in all_notes]
    raw_durations = [n[2] for n in all_notes]
    mean_pitch = np.mean(raw_pitches)
    mean_duration = np.mean(raw_durations)

    numeric_features_vec = np.array([
        tempo,
        key_num,
        mean_pitch,
        mean_duration,
        len(note_array), # Number of notes
        0.0 # Placeholder
    ], dtype=np.float32)

    # Pad / Truncate Note Array
    if len(note_array) > max_notes_const:
        note_array_fixed = note_array[:max_notes_const, :]
    else:
        pad_width = max_notes_const - len(note_array)
        note_array_fixed = np.pad(
            note_array,
            pad_width=((0, pad_width), (0, 0)),
            mode='constant',
            constant_values=PAD_VALUE
        )

    return note_array_fixed, numeric_features_vec

# --- 3. Main Preprocessing Loop ---
print(f"Loading manifest from {MANIFEST_FILE}...")
manifest = pd.read_csv(MANIFEST_FILE)

print(f"Starting preprocessing for {len(manifest)} files...")
for _, row in tqdm(manifest.iterrows(), total=len(manifest)):

    features = extract_features(row['full_path'], MAX_NOTES)

    if features:
        note_array, numeric_features = features

        original_stem = Path(row['full_path']).stem
        sanitized_stem = sanitize_filename(original_stem)
        output_filename = f"{row['source']}_{sanitized_stem}.npz"
        output_path = PROCESSED_PATH / output_filename

        np.savez_compressed(
            output_path,
            notes=note_array,
            numeric_features=numeric_features,
            emotion=row['emotion'],
            source=row['source']
        )
    else:
        # print(f"Skipped (likely corrupt): {row['full_path']}")
        pass

print("\n--- Preprocessing Complete ---")
print(f"All .npz files in {PROCESSED_PATH} are now correct.")