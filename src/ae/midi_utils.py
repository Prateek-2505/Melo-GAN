# src/ae/midi_utils.py
import os
import numpy as np
from .path_utils import ensure_dir # Import from your new utils file

try:
    import pretty_midi
except Exception:
    pretty_midi = None
    print("pretty_midi not installed; MIDI export disabled. Install pretty_midi to enable MIDI reconstructions.")

def notes_array_to_prettymidi(notes_arr: np.ndarray, tempo: float = 120.0, instrument_program: int = 0):
    """
    Convert notes array (MAX_NOTES,4) to PrettyMIDI object.
    Notes array columns: pitch, start_rel, duration, velocity
    Only non-zero pitch rows are converted.
    """
    if pretty_midi is None:
        return None
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=instrument_program)
    # notes may be padded rows of zeros; filter
    for row in notes_arr:
        p, s, d, v = row
        if p <= 0 or d <= 0:
            continue
        start = float(s)
        end = float(s + d)
        # clamp pitch to MIDI range
        pitch = int(np.clip(round(p), 0, 127))
        vel = int(np.clip(round(v), 1, 127))
        note = pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)
    pm.instruments.append(instrument)
    return pm

def save_recon_midi(notes_in: np.ndarray, notes_out: np.ndarray, outdir: str, prefix: str, tempo=120.0):
    ensure_dir(outdir)
    if pretty_midi is None:
        return
    pm_in = notes_array_to_prettymidi(notes_in, tempo=tempo)
    pm_out = notes_array_to_prettymidi(notes_out, tempo=tempo)
    if pm_in:
        pm_in.write(os.path.join(outdir, f"{prefix}_in.mid"))
    if pm_out:
        pm_out.write(os.path.join(outdir, f"{prefix}_out.mid"))