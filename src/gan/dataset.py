"""
Robust dataset loader for GAN training.
- Now loads 'numeric_features' from .npz files or <split>/numeric_features.npy
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class GANDataset(Dataset):
    def __init__(self, split_csv, processed_dir="data/processed", 
                 notes_npy=None, emotion_npy=None, latent_feats=None, 
                 numeric_features_npy=None, numeric_input_dim=6,latent_dim=128):
        """
        split_csv: path to csv with manifest
        processed_dir: directory with per-file .npz produced by preprocessing
        notes_npy/emotion_npy/latent_feats/numeric_features_npy: optional pre-saved arrays
        numeric_input_dim: fallback dimension for numeric features if missing
        """
        self.processed_dir = processed_dir
        self.split_csv = split_csv
        self.numeric_input_dim = numeric_input_dim
        self.latent_dim = latent_dim  
        df = pd.read_csv(split_csv)
        self.df = df

        # --- Fast NPY Path ---
        # Check if all required .npy files exist
        if (notes_npy and os.path.exists(notes_npy) and
            emotion_npy and os.path.exists(emotion_npy) and
            numeric_features_npy and os.path.exists(numeric_features_npy)):
            
            print(f"[INFO] Loading data from pre-saved NPY files for split: {Path(notes_npy).parent.name}")
            self.notes = np.load(notes_npy)
            self.emotions = np.load(emotion_npy)
            self.numeric_features = np.load(numeric_features_npy)
            self.use_npy = True

            # Ensure alignment
            n_samples = self.notes.shape[0]
            if not (self.emotions.shape[0] == n_samples and self.numeric_features.shape[0] == n_samples):
                raise ValueError("NPY file length mismatch (notes, emotions, numeric_features)")

            if latent_feats is not None:
                if latent_feats.shape[0] == n_samples:
                    self.latent_feats = latent_feats
                else:
                    print(f"[WARN] latent_feats length mismatch ({latent_feats.shape[0]}) vs notes ({n_samples}). Ignoring latent_feats.")
                    self.latent_feats = None
            else:
                self.latent_feats = None
            
            return

        # --- Slow .npz Path ---
        print(f"[INFO] Loading data by reading individual .npz files for split: {Path(split_csv).stem}")
        self.use_npy = False
        self.processed_paths = []
        self.moods = []
        self.latent_feats_list = [] # Only used if latent_feats array is passed
        self.numeric_features_list = [] # Stores numeric features loaded from .npz

        # Determine file identifier column
        preferred_cols = ['npz_path', 'processed_file', 'processed', 'full_path', 'filepath', 'file', 'filename', 'file_key']
        self.col_found = None
        for c in preferred_cols:
            if c in df.columns:
                self.col_found = c
                break
        if self.col_found is None:
            raise KeyError(f"Split CSV must contain one of {preferred_cols}. Found columns: {list(df.columns)}")

        # Build paths and load metadata
        for idx, r in df.iterrows():
            raw_cell = r[self.col_found]
            resolved = self._resolve_npz_path(raw_cell, r)

            if resolved is None:
                print(f"[WARN] Could not find processed .npz for manifest row: {dict(r)}")
                continue

            self.processed_paths.append(resolved)
            
            # Try to get mood/emotion
            mood = None
            for col in ['emotion', 'mood', 'label']:
                if col in df.columns:
                    mood = r[col]
                    break
            self.moods.append(mood)
            
            # Pre-load numeric features if we are in .npz mode
            try:
                with np.load(resolved, allow_pickle=True) as data:
                    default_numeric = np.zeros(self.numeric_input_dim, dtype='float32')
                    numeric = data.get('numeric_features', default_numeric).astype('float32')
                    # Handle potential shape mismatches
                    if numeric.ndim == 0 or numeric.size == 0:
                        numeric = default_numeric
                    elif numeric.size != self.numeric_input_dim:
                         # Pad or truncate
                        new_numeric = default_numeric
                        copy_len = min(numeric.size, self.numeric_input_dim)
                        new_numeric[:copy_len] = numeric.flatten()[:copy_len]
                        numeric = new_numeric
                    
                    self.numeric_features_list.append(numeric)
            except Exception as e:
                print(f"[ERROR] Failed to load numeric_features from {resolved}: {e}")
                self.numeric_features_list.append(np.zeros(self.numeric_input_dim, dtype='float32'))


        # Align provided latent_feats array (if any)
        if latent_feats is not None:
            if len(latent_feats) == len(self.processed_paths):
                self.latent_feats = latent_feats # Store the whole array
            else:
                print("[WARN] provided latent_feats length does not match resolved split size; ignoring.")
                self.latent_feats = None
        else:
            self.latent_feats = None


    def _resolve_npz_path(self, raw_cell, row_data):
        """Helper to find the .npz file path from manifest info."""
        if not isinstance(raw_cell, str):
            raw_cell = str(raw_cell)

        # 1) if it's already a valid path (relative or absolute)
        candidate = raw_cell if os.path.isabs(raw_cell) else os.path.join(self.processed_dir, raw_cell)
        if raw_cell.lower().endswith('.npz') and os.path.exists(candidate):
            return candidate

        # 2) if it's an absolute/relative path to original midi (full_path)
        if os.path.exists(str(raw_cell)):
            stem = os.path.splitext(os.path.basename(raw_cell))[0]
            cand = glob.glob(os.path.join(self.processed_dir, f"*{stem}*.npz"))
            if cand:
                return cand[0]

        # 3) try by stem search using the raw cell's basename
        stem = os.path.splitext(os.path.basename(raw_cell))[0]
        cand = glob.glob(os.path.join(self.processed_dir, f"*{stem}*.npz"))
        if cand:
            return cand[0]

        # 4) fallback: if manifest contains 'npz_path' column
        if 'npz_path' in row_data:
            alt = row_data.get('npz_path', '')
            if isinstance(alt, str) and alt:
                candidate = alt if os.path.isabs(alt) else os.path.join(self.processed_dir, alt)
                if os.path.exists(candidate):
                    return candidate
        return None

    def __len__(self):
        if self.use_npy:
            return len(self.notes)
        else:
            return len(self.processed_paths)

    def __getitem__(self, idx):
        if self.use_npy:
            notes = self.notes[idx].astype('float32')
            emotion = self.emotions[idx]
            if self.latent_feats is not None:
                latent = self.latent_feats[idx].astype('float32')
            else:
                latent = np.zeros(self.latent_dim, dtype='float32')
            numeric = self.numeric_features[idx].astype('float32')
            return notes, emotion, latent, numeric
        else:
            path = self.processed_paths[idx]
            data = np.load(path, allow_pickle=True)
            
            notes = data['notes'].astype('float32')
            
            # Mood: Check .npz, then manifest
            mood = data.get('mood', None)
            if mood is None:
                mood = self.moods[idx] if idx < len(self.moods) else None
                
            # Latent (from pre-passed array)
            latent_val = self.latent_feats[idx] if self.latent_feats is not None and idx < len(self.latent_feats) else None
            if latent_val is not None:
                latent = latent_val.astype('float32')
            else:
                latent = np.zeros(self.latent_dim, dtype='float32')
            
            # Numeric (pre-loaded in __init__)
            numeric = self.numeric_features_list[idx]
            
            return notes, mood, latent, numeric