# src/ae/dataset.py
import os
import random
import numpy as np
from typing import List, Dict
import torch
from torch.utils.data import Dataset

# --- Augmentation Helpers ---

def augment_tempo_scale(notes: np.ndarray, scale: float) -> np.ndarray:
    # notes[:,1] = start_rel, notes[:,2] = duration
    out = notes.copy()
    out[:,1] = out[:,1] * scale
    out[:,2] = out[:,2] * scale
    return out

def augment_pitch_shift(notes: np.ndarray, semitones: int) -> np.ndarray:
    out = notes.copy()
    out[:,0] = out[:,0] + semitones
    return out

def augment_note_dropout(notes: np.ndarray, dropout_prob: float) -> np.ndarray:
    out = notes.copy()
    if dropout_prob <= 0:
        return out
    mask = np.random.rand(out.shape[0]) > dropout_prob
    out[~mask, :] = 0.0
    return out

def augment_velocity_jitter(notes: np.ndarray, jitter: float) -> np.ndarray:
    out = notes.copy()
    out[:,3] = out[:,3] + np.random.normal(0, jitter, size=out.shape[0])
    return out

def timing_jitter(notes: np.ndarray, sigma_seconds: float) -> np.ndarray:
    out = notes.copy()
    out[:,1] = out[:,1] + np.random.normal(0, sigma_seconds, size=out.shape[0])
    out[:,1] = np.clip(out[:,1], a_min=0.0, a_max=None)
    return out

# --- Dataset Class ---

class MIDIDataset(Dataset):
    """
    Loads processed .npz files (notes array).
    expected file structure: processed/*.npz
    Each file contains 'notes' array shaped (MAX_NOTES,4) and keys 'tempo' and 'filename'
    """
    def __init__(self, file_list: List[str], config: Dict, augment: bool = False):
        self.files = file_list
        self.augment = augment
        self.cfg = config
        self.max_notes = config['MAX_NOTES']
        # augmentation params
        self.tempo_jitter = config['AUGMENT']['tempo_jitter']
        self.pitch_shift = config['AUGMENT']['pitch_shift']
        self.note_dropout = config['AUGMENT']['note_dropout']
        self.velocity_jitter = config['AUGMENT']['velocity_jitter']
        self.timing_jitter = config['AUGMENT']['timing_jitter']

    def __len__(self):
        return len(self.files)

# ... (inside your MIDIDataset class in src/ae/dataset.py) ...
    
    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        notes = data['notes'].astype(np.float32)  # (MAX_NOTES, 4)

        # --- NEW: NORMALIZE THE DATA ---
        # Create a mask to avoid scaling our -1 padding
        mask = (notes[:, 0] != -1)

        # 1. Scale Pitch (0-127) -> (-1, 1)
        # We use 128.0 to get a range of [-1, 0.99]
        notes[mask, 0] = (notes[mask, 0] / 128.0) * 2.0 - 1.0
        
        # 2. Scale Velocity (0-127) -> (-1, 1)
        # Clip at 127 in case of bad data (like your 194.0)
        notes[mask, 3] = np.clip(notes[mask, 3], 0, 127)
        notes[mask, 3] = (notes[mask, 3] / 128.0) * 2.0 - 1.0

        # 3. Scale Beats (Start & Duration)
        #    We scale them to [0, 1] based on a large max.
        notes[mask, 1] = notes[mask, 1] / self.cfg.get('MAX_START_BEAT', 100.0)
        notes[mask, 2] = notes[mask, 2] / self.cfg.get('MAX_DURATION_BEAT', 20.0)
        # --- End of Normalization ---
        
 # augmentations (probabilistic)
        if self.augment:
            if random.random() < 0.3 and self.tempo_jitter > 0:
                scale = 1.0 + random.uniform(-self.tempo_jitter, self.tempo_jitter)
                notes = augment_tempo_scale(notes, scale)
            if random.random() < 0.3 and self.pitch_shift != 0:
                shift = random.randint(-self.pitch_shift, self.pitch_shift)
                notes = augment_pitch_shift(notes, shift)
            if random.random() < 0.2 and self.note_dropout > 0:
                notes = augment_note_dropout(notes, self.note_dropout)
            if random.random() < 0.3 and self.velocity_jitter > 0:
                notes = augment_velocity_jitter(notes, self.velocity_jitter)
            if random.random() < 0.2 and self.timing_jitter > 0:
                notes = timing_jitter(notes, self.timing_jitter)
        
        notes = np.nan_to_num(notes, nan=0.0, posinf=0.0, neginf=0.0)
        notes = notes.astype(np.float32)
        
        return notes, str(data.get('filename',''))