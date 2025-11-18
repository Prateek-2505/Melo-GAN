#!/usr/bin/env python3
"""
Analyze MIDI files to extract musical features.
Useful for checking if the GAN is actually learning different emotions.
"""

import os
import argparse
import numpy as np
import pretty_midi

def analyze_file(filepath):
    try:
        pm = pretty_midi.PrettyMIDI(filepath)
        
        if len(pm.instruments) == 0:
            print(f"[-] {filepath}: No instruments found.")
            return

        # We assume the piano is the first instrument
        inst = pm.instruments[0]
        notes = inst.notes
        
        if len(notes) == 0:
            print(f"[-] {filepath}: No notes found.")
            return

        # --- Extract Statistics ---
        total_duration = pm.get_end_time()
        
        pitches = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])
        durations = np.array([n.end - n.start for n in notes])
        
        # 1. Pitch Stats
        avg_pitch = np.mean(pitches)
        min_pitch = np.min(pitches)
        max_pitch = np.max(pitches)
        unique_pitches = len(np.unique(pitches))
        
        # 2. Velocity (Volume) Stats
        avg_velocity = np.mean(velocities)
        
        # 3. Rhythm/Density Stats
        notes_per_second = len(notes) / total_duration if total_duration > 0 else 0
        
        print(f"analysis for: {os.path.basename(filepath)}")
        print(f"  Duration:     {total_duration:.2f} seconds")
        print(f"  Note Count:   {len(notes)}")
        print(f"  Avg Pitch:    {avg_pitch:.2f} (MIDI Note Number)")
        print(f"  Pitch Range:  {min_pitch} - {max_pitch}")
        print(f"  Unique Notes: {unique_pitches} (Variety check)")
        print(f"  Avg Velocity: {avg_velocity:.2f} (Volume)")
        print(f"  Density:      {notes_per_second:.2f} notes/sec")
        print("-" * 40)
        
    except Exception as e:
        print(f"[ERROR] Could not analyze {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="List of .mid files to analyze")
    args = parser.parse_args()
    
    print("="*40)
    print("      MIDI ANALYSIS REPORT")
    print("="*40)
    
    for f in args.files:
        if os.path.exists(f):
            analyze_file(f)
        else:
            print(f"[WARN] File not found: {f}")