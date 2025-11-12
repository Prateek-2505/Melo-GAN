"""
Robust dataset loader for GAN training.

- Accepts split CSVs that may use columns:
  'npz_filename', 'processed_file', 'processed', 'full_path', 'filepath',
  'file', 'filename', 'file_key'
- If precomputed notes.npy & emotion.npy exist under <SPLITS_DIR>/<split>/, will load them.
- If latent_feats is provided (numpy array), it will be used as returned latent per-sample.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class GANDataset(Dataset):
    def __init__(self, split_csv, processed_dir="data/processed", notes_npy=None, emotion_npy=None, latent_feats=None):
        """
        split_csv: path to csv with manifest (any column naming convention handled)
        processed_dir: directory with per-file .npz produced by preprocessing
        notes_npy/emotion_npy: optional pre-saved arrays under splits/<split>/
        latent_feats: optional numpy array aligned with split (N, LATENT_DIM)
        """
        self.processed_dir = processed_dir
        self.split_csv = split_csv
        df = pd.read_csv(split_csv)
        self.df = df  # keep for debugging

        # If notes_npy and emotion_npy are valid files, load them for fast access
        if notes_npy and os.path.exists(notes_npy) and emotion_npy and os.path.exists(emotion_npy):
            self.notes = np.load(notes_npy)
            self.emotions = np.load(emotion_npy)
            self.use_npy = True
            # If latent_feats provided, ensure alignment
            self.latent_feats = latent_feats if latent_feats is None or latent_feats.shape[0] == self.notes.shape[0] else None
            return

        self.use_npy = False
        # Determine which column in CSV holds the identifier for file lookup
        preferred_cols = ['npz_filename', 'processed_file', 'processed', 'full_path', 'filepath', 'file', 'filename', 'file_key']
        self.col_found = None
        for c in preferred_cols:
            if c in df.columns:
                self.col_found = c
                break
        if self.col_found is None:
            raise KeyError(f"Split CSV must contain one of {preferred_cols}. Found columns: {list(df.columns)}")

        # Build processed paths list by resolving each manifest row to a processed .npz
        self.processed_paths = []
        self.moods = []  # will store the emotion column if present (string)
        for _, r in df.iterrows():
            raw_cell = r[self.col_found]
            resolved = None

            # 1) if npz_filename-like already points to .npz (relative or absolute)
            if isinstance(raw_cell, str) and raw_cell.lower().endswith('.npz'):
                candidate = raw_cell if os.path.isabs(raw_cell) else os.path.join(self.processed_dir, raw_cell)
                if os.path.exists(candidate):
                    resolved = candidate

            # 2) if it's an absolute/relative path to original midi (full_path)
            if resolved is None and isinstance(raw_cell, str) and os.path.exists(str(raw_cell)):
                # try to find a processed .npz with same stem
                stem = os.path.splitext(os.path.basename(raw_cell))[0]
                cand = glob.glob(os.path.join(self.processed_dir, f"*{stem}*.npz"))
                if cand:
                    resolved = cand[0]

            # 3) try by stem search using the raw cell's basename
            if resolved is None:
                stem = os.path.splitext(os.path.basename(str(raw_cell)))[0]
                cand = glob.glob(os.path.join(self.processed_dir, f"*{stem}*.npz"))
                if cand:
                    resolved = cand[0]

            # 4) fallback: if manifest contains 'npz_filename' column, try that specifically
            if resolved is None and 'npz_filename' in df.columns:
                alt = r.get('npz_filename', '')
                if isinstance(alt, str) and alt:
                    candidate = alt if os.path.isabs(alt) else os.path.join(self.processed_dir, alt)
                    if os.path.exists(candidate):
                        resolved = candidate

            if resolved is None:
                # could not resolve; warn and skip this row
                print("Warning: could not find processed .npz for manifest row:", dict(r))
                continue

            self.processed_paths.append(resolved)
            # try to get mood/emotion column if exists
            mood = None
            for col in ['emotion', 'mood', 'label']:
                if col in df.columns:
                    mood = r[col]
                    break
            self.moods.append(mood)

        # latent_feats alignment: if provided, require lengths match the dataset (otherwise ignore)
        self.latent_feats = latent_feats if latent_feats is None or len(latent_feats) == len(self.processed_paths) else None
        if latent_feats is not None and self.latent_feats is None:
            print("Warning: provided latent_feats length does not match resolved split size; ignoring latent_feats.")

    def __len__(self):
        if self.use_npy:
            return len(self.notes)
        else:
            return len(self.processed_paths)

    def __getitem__(self, idx):
        if self.use_npy:
            notes = self.notes[idx].astype('float32')
            emotion = self.emotions[idx]
            latent = self.latent_feats[idx].astype('float32') if self.latent_feats is not None else None
            return notes, emotion, latent
        else:
            path = self.processed_paths[idx]
            data = np.load(path, allow_pickle=True)
            notes = data['notes'].astype('float32')
            # mood may be stored inside npz for labeled examples
            mood = data.get('mood', None)
            # fallback to manifest moods collected earlier
            if mood is None:
                mood = self.moods[idx] if idx < len(self.moods) else None
            latent = self.latent_feats[idx].astype('float32') if self.latent_feats is not None else None
            return notes, mood, latent
