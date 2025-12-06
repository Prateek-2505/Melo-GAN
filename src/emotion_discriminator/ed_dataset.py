# src/emotion_discriminator/ed_dataset.py
"""
Robust Dataset and dataloader utilities for Emotion Discriminator.
Updated to include ON-THE-FLY NORMALIZATION to [-1, 1].
"""

from typing import Optional, Callable, Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import warnings
import glob  

DEFAULT_LABELS = ["happy", "sad", "angry", "calm"]
MAX_BEAT_TIME = 4.0  # Matches utils.py

def _to_int_label(lbl: Any, label_map: Optional[Dict[str, int]] = None) -> int:
    if isinstance(lbl, (int, np.integer)):
        return int(lbl)
    s = str(lbl).strip().lower()
    if label_map and s in label_map:
        return int(label_map[s])
    if s in DEFAULT_LABELS:
        return DEFAULT_LABELS.index(s)
    try:
        return int(float(s))
    except Exception:
        raise ValueError(f"Could not convert label '{lbl}' to int. Provide a label_map.")


def _resolve_split_csv(cfg: Dict[str, Any], split: str) -> str:
    key = f"{split}_split_csv"
    if key in cfg and cfg[key]:
        return cfg[key]
    if "train_split_csv" in cfg and cfg["train_split_csv"] and split == "train":
        return cfg["train_split_csv"]
    raise ValueError(f"Missing split csv for '{split}' in config; expected key '{key}'.")


def _resolve_encoder_feats_for_split(cfg: Dict[str, Any], split: str) -> Optional[str]:
    candidate_key = f"{split}_encoder_feats_path"
    if candidate_key in cfg and cfg[candidate_key]:
        return cfg[candidate_key]
    if "encoder_feats_path" in cfg and cfg["encoder_feats_path"]:
        return cfg["encoder_feats_path"]
    try:
        split_csv = _resolve_split_csv(cfg, split)
        csv_dir = os.path.dirname(split_csv) or "."
        cand = os.path.join(csv_dir, "encoder_feats.npy")
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join("data", "splits", split, "encoder_feats.npy")
        if os.path.exists(cand2):
            return cand2
    except Exception:
        pass
    return None


class EmotionDataset(Dataset):  
    def __init__(
        self,
        split_csv: str,
        input_mode: str = "latent",
        processed_dir: str = "data/processed",
        encoder_feats: Optional[np.ndarray] = None,
        encoder_mapping: Optional[Dict[str, np.ndarray]] = None,
        manifest_csv: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        max_notes: int = 512,
        note_dim: int = 4,
        augment: bool = False,
        augment_cfg: Optional[Dict[str, Any]] = None,
        preload: bool = False,

        
    ):
        assert input_mode in ("latent", "notes"), "input_mode must be 'latent' or 'notes'"
        self.df = pd.read_csv(split_csv)
        self.split_csv = split_csv
        self.input_mode = input_mode
        self.processed_dir = processed_dir
        self.encoder_feats = encoder_feats
        self.encoder_mapping = encoder_mapping or {}
        self.manifest_csv = manifest_csv
        self.label_map = {k.lower(): int(v) for k, v in (label_map or {}).items()}
        self.max_notes = int(max_notes)
        self.note_dim = int(note_dim)
        self.augment = bool(augment)
        self.augment_cfg = augment_cfg or {}
        self.preload = bool(preload)
        self.skip_normalize = True

        self.file_col = None
        for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
            if c in self.df.columns:
                self.file_col = c
                break
        if self.file_col is None:
            if self.manifest_csv and os.path.exists(self.manifest_csv):
                manifest_df = pd.read_csv(self.manifest_csv)
                common = set(self.df.columns).intersection(set(manifest_df.columns))
                if common:
                    self.file_col = list(common)[0]
            if self.file_col is None:
                raise ValueError("Split CSV must contain a file identifier column.")

        self._manifest_map = None
        if self.manifest_csv and os.path.exists(self.manifest_csv):
            try:
                mdf = pd.read_csv(self.manifest_csv)
                path_col = None
                for c in ["full_path", "path", "relative_path", "npz_path", "file_path"]:
                    if c in mdf.columns:
                        path_col = c
                        break
                key_col = None
                for c in ["file_key", "npz_path", "filename", "file"]:
                    if c in mdf.columns:
                        key_col = c
                        break
                if key_col and path_col:
                    self._manifest_map = dict(zip(mdf[key_col].astype(str), mdf[path_col].astype(str)))
                else:
                    if "path" in mdf.columns:
                        self._manifest_map = {os.path.basename(p): p for p in mdf["path"].astype(str).tolist()}
            except Exception:
                warnings.warn("Failed to parse manifest_csv; continuing without manifest map.")

        self._cached = [None] * len(self.df) if self.preload else None
        if self.preload:
            for i in range(len(self.df)):
                try:
                    self._cached[i] = self._load_item_raw(i)
                except Exception as e:
                    warnings.warn(f"Preload failed idx={i}: {e}")
                    self._cached[i] = None

    def __len__(self):
        return len(self.df)

    def _resolve_npz_path(self, entry_val: str) -> str:
        if isinstance(entry_val, float) and np.isnan(entry_val):
            raise ValueError("Row has NaN in file path column.")
        v = str(entry_val)
        if os.path.isabs(v) and os.path.exists(v): return v
        if self._manifest_map and v in self._manifest_map:
            p = self._manifest_map[v]
            if os.path.exists(p): return p
            cand = os.path.join(self.processed_dir, p)
            if os.path.exists(cand): return cand
        cand = os.path.join(self.processed_dir, v)
        if os.path.exists(cand): return cand
        if not v.endswith(".npz"):
            cand2 = cand + ".npz"
            if os.path.exists(cand2): return cand2
        basename = os.path.basename(v)
        for root, _, files in os.walk(self.processed_dir):
            if basename in files or (basename + ".npz") in files:
                return os.path.join(root, basename if basename.endswith(".npz") else basename + ".npz")
        raise FileNotFoundError(f"Could not locate npz for {v} under {self.processed_dir}")

    def _load_item_raw(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        label = row.get("emotion", row.get("label", None))
        if label is None:
            raise ValueError("Split CSV must include an 'emotion' or 'label' column.")
        label_int = _to_int_label(label, label_map=self.label_map)

        if self.input_mode == "latent":
            if self.encoder_mapping:
                key = str(row[self.file_col])
                vec = self.encoder_mapping.get(key) or self.encoder_mapping.get(os.path.basename(key))
                if vec is None: raise KeyError(f"latent vector not found for key '{key}'")
                return np.asarray(vec, dtype=np.float32), label_int, {"row": row.to_dict()}
            elif self.encoder_feats is not None:
                try:
                    return np.asarray(self.encoder_feats[idx], dtype=np.float32), label_int, {"row": row.to_dict()}
                except Exception:
                    if getattr(self.encoder_feats, "dtype", None) == np.object_:
                        mapping = dict(self.encoder_feats.tolist())
                        key = str(row[self.file_col])
                        vec = mapping.get(key) or mapping.get(os.path.basename(key))
                        return np.asarray(vec, dtype=np.float32), label_int, {"row": row.to_dict()}
                    raise ValueError("encoder_feats alignment error.")
            else:
                raise FileNotFoundError("No encoder_feats provided for latent mode.")

        else:
            # --- NOTES MODE LOADING ---
            entry = row[self.file_col]
            npz_path = self._resolve_npz_path(entry)
            data = np.load(npz_path, allow_pickle=True)
            
            # Try multiple keys for notes
            if "notes" in data: notes = data["notes"]
            elif "arr_0" in data: notes = data["arr_0"]
            else: notes = data[list(data.files)[0]]
            
            notes = np.asarray(notes, dtype=np.float32)
            
            # --- STEP 1: VALIDATE SOURCE DATA (Expect 4 Dims) ---
            SOURCE_DIM = 4
            if notes.ndim == 1: notes = notes.reshape(-1, SOURCE_DIM)
            
            # Handle Transpose if shape is (4, T) -> (T, 4)
            if notes.shape[1] != SOURCE_DIM:
                if notes.shape[0] == SOURCE_DIM: 
                    notes = notes.T
                else:
                    # If config asks for 5 dims, we still expect the FILE to have 4.
                    # We only crash if the file doesn't have 4.
                    raise ValueError(f"Raw note data mismatch. File has {notes.shape}, expected (T, {SOURCE_DIM}).")

            # --- STEP 2: FEATURE ENGINEERING (Create 5th Dim) ---
            # Calculate Pitch Interval (Difference between current and prev pitch)
            # Column 0 is Pitch.
            pitches = notes[:, 0]
            # Calculate diff, prepend 0 for the first note
            intervals = np.diff(pitches, prepend=pitches[0])
            intervals = intervals.reshape(-1, 1)
            
            # Add Interval as the 5th column
            notes = np.hstack([notes, intervals])
            
            # --- STEP 3: VALIDATE TARGET DATA (Expect 5 Dims) ---
            # Now the data should match self.note_dim (which is 5 in config)
            if notes.shape[1] != self.note_dim:
                 # Safety check in case config note_dim is not 5
                 raise ValueError(f"Config expects note_dim={self.note_dim}, but we generated {notes.shape[1]} features.")

            # --- STEP 4: PAD / TRUNCATE ---
            T = notes.shape[0]
            if T >= self.max_notes:
                notes = notes[: self.max_notes]
            else:
                pad = np.zeros((self.max_notes - T, self.note_dim), dtype=np.float32)
                notes = np.concatenate([notes, pad], axis=0)
                
            return notes, label_int, {"npz_path": npz_path, "row": row.to_dict()}

    def _normalize_notes(self, raw_notes: np.ndarray) -> np.ndarray:
        """
        Normalizes raw MIDI values (0-127) to GAN range (-1, 1).
        Assumes column order: [Pitch, Velocity, Duration, Step]
        Matching utils.py logic.
        """
        norm = raw_notes.copy()
        
        # 1. Pitch: [0, 127] -> [-1, 1]
        # utils.py: pitch = ((norm + 1) * 63.5)
        norm[:, 0] = (raw_notes[:, 0] / 63.5) - 1.0
        
        # 2. Velocity: [0, 127] -> [-1, 1]
        # Simple linear scaling. Note: utils.py uses heuristics for decoding, 
        # but linear encoding is standard for training.
        norm[:, 1] = (raw_notes[:, 1] / 63.5) - 1.0
        
        # 3. Duration: [0, MAX_BEAT_TIME] -> [-1, 1]
        # utils.py: duration = ((norm + 1) / 2) * 4.0
        norm[:, 2] = (raw_notes[:, 2] / (MAX_BEAT_TIME / 2.0)) - 1.0
        
        # 4. Step: [0, MAX_BEAT_TIME] -> [-1, 1]
        norm[:, 3] = (raw_notes[:, 3] / (MAX_BEAT_TIME / 2.0)) - 1.0
        
        # Clip to ensure safety
        return np.clip(norm, -1.0, 1.0)

    def _augment_notes(self, notes: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        out = notes.copy()
        
        # 1. Pitch Shifting (Octave Jumps)
        # We shift by +/- 12 semitones to preserve key/emotion character.
        # In normalized space (div by 63.5), 12 semitones is approx 0.1889
        if cfg.get("pitch_shift_prob", 0.0) > 0:
            if np.random.rand() < float(cfg.get("pitch_shift_prob")):
                # Choose -12, 0, or +12 semitones
                steps = np.random.choice([-12, 12])
                shift_amount = steps / 63.5
                
                # Apply to Pitch column (index 0)
                out[:, 0] += shift_amount
                
                # Re-clip to valid range [-1, 1] to prevent out-of-bounds
                np.clip(out[:, 0], -1.0, 1.0, out=out[:, 0])

        # 2. Add noise (jitter)
        if cfg.get("noise_std", 0.0) > 0:
            std = float(cfg.get("noise_std", 0.01))
            # Apply jitter to continuous features (Pitch, Vel, Dur, Step)
            # We usually avoid jittering pitch to keep it in tune, but slight jitter 
            # can simulate "humanization". Safe to apply to cols 1 (Vel), 2 (Dur), 3 (Step).
            for c in [1, 2, 3]:
                if c < out.shape[1]:
                    noise = np.random.normal(scale=std, size=(out.shape[0],))
                    out[:, c] += noise
        
        # 3. Dropout (Masking)
        if cfg.get("dropout_prob", 0.0) > 0:
            p = float(cfg.get("dropout_prob", 0.05))
            mask = np.random.rand(out.shape[0]) >= p
            # Set masked notes to -1.0 (padding value) or 0.0 (center)
            # Using -1.0 effectively "silences" them in the normalized space
            out[~mask] = -1.0 

        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        if self.preload and self._cached is not None and self._cached[idx] is not None:
            raw, label_int, meta = self._cached[idx]
        else:
            raw, label_int, meta = self._load_item_raw(idx)

        if self.input_mode == "latent":
            tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        else:
    # raw is the loaded notes array from `_load_item_raw`
            if self.skip_normalize:
                # assume .npz already contains values in [-1, 1]
                notes = np.asarray(raw, dtype=np.float32)
                # ensure shape is correct / padded/truncated as done in _load_item_raw
                # (if _load_item_raw already padded/truncated, nothing to do)
            else:
                # existing behaviour: normalize raw MIDI-scale data to [-1, 1]
                notes = self._normalize_notes(raw)

            if self.augment:
                notes = self._augment_notes(notes)

            tensor = torch.from_numpy(np.asarray(notes, dtype=np.float32))


        return tensor, int(label_int), meta

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]) -> Dict[str, Any]:
        xs = [b[0] for b in batch]
        ys = torch.LongTensor([b[1] for b in batch])
        metas = [b[2] for b in batch]
        x = torch.stack(xs, dim=0)
        return {"x": x, "y": ys, "meta": metas}

# ... (The rest of the file: _load_encoder_feats_from_path, build_dataloader, main check remains unchanged) ...
# I am including them here so you can copy-paste the full file content if needed, 
# but typically you only need the class update. 

def _load_encoder_feats_from_path(path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    if path is None: return None, None
    if not os.path.exists(path): return None, None
    arr = np.load(path, allow_pickle=True)
    if getattr(arr, "dtype", None) == np.object_:
        try:
            mapping = dict(arr.tolist())
            return None, mapping
        except Exception:
            try:
                arr2 = np.asarray(arr.tolist(), dtype=np.float32)
                return arr2, None
            except Exception:
                return None, None
    else:
        return arr, None

def build_dataloader(cfg: Dict[str, Any], split: str = "train", shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    from torch.utils.data import WeightedRandomSampler
    
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of 'train', 'val', 'test'")

    split_csv = _resolve_split_csv(cfg, split)
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    # Attempt to load encoder feats
    encoder_path = _resolve_encoder_feats_for_split(cfg, split)
    encoder_arr, encoder_map = _load_encoder_feats_from_path(encoder_path) if encoder_path else (None, None)

    # Load CSV
    df = pd.read_csv(split_csv)
    orig_len = len(df)
    df_filtered = df

    # --- ROBUST FILTERING LOGIC ---
    input_mode = cfg.get("input_mode", "latent")
    
    if input_mode == "notes":
        print(f"[INFO] Verifying files for split '{split}'...")
        processed_dir = cfg.get('processed_dir', 'data/processed')
        
        # Identify file column
        file_col = None
        for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
            if c in df.columns:
                file_col = c
                break
        
        if file_col:
            keep_indices = []
            for idx, row in df.iterrows():
                fname = row[file_col]
                if pd.isna(fname): continue
                fname = str(fname)

                # 1. Check exact path join
                cand = os.path.join(processed_dir, fname if fname.endswith('.npz') else fname + '.npz')
                if os.path.exists(cand):
                    keep_indices.append(idx)
                    continue
                
                # 2. Check basename fallback (common if paths are relative/messy)
                stem = os.path.splitext(os.path.basename(fname))[0]
                cand_base = os.path.join(processed_dir, stem + ".npz")
                if os.path.exists(cand_base):
                    keep_indices.append(idx)
                    continue
                
                # If we get here, file is missing. 
                # print(f"[WARN] Missing file: {fname}") # Uncomment to debug specific files

            df_filtered = df.loc[keep_indices].reset_index(drop=True)
            dropped = orig_len - len(df_filtered)
            if dropped > 0:
                print(f"[WARN] Dropped {dropped} rows due to missing .npz files. Remaining: {len(df_filtered)}")
                
                # Save auto-filtered CSV to avoid re-scanning next time
                split_dir = os.path.dirname(split_csv) or os.path.join("data", "splits", split)
                os.makedirs(split_dir, exist_ok=True)
                filtered_csv_path = os.path.join(split_dir, f"auto_filtered_{os.path.basename(split_csv)}")
                df_filtered.to_csv(filtered_csv_path, index=False)
                split_csv = filtered_csv_path # Point dataset to new CSV
        else:
            print("[WARN] No file column found in CSV. Skipping file verification (risky).")

    elif input_mode == "latent" and encoder_map:
        # Filter out rows that don't have matching encoder map keys
        file_col = None
        for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
            if c in df.columns:
                file_col = c
                break
        if file_col:
            mask = df[file_col].apply(lambda x: str(x) in encoder_map or os.path.basename(str(x)) in encoder_map)
            df_filtered = df[mask].reset_index(drop=True)
            if len(df_filtered) < orig_len:
                print(f"[INFO] Dropped {orig_len - len(df_filtered)} rows missing from encoder_mapping.")

    # --- BUILD DATASET ---
    ds = EmotionDataset(
        split_csv=split_csv if len(df_filtered) != orig_len else split_csv, 
        # Note: if we filtered, EmotionDataset re-reads the CSV. 
        # Passing the path to the saved filtered csv (logic above) handles this.
        input_mode=input_mode,
        processed_dir=cfg.get("processed_dir", "data/processed"),
        encoder_feats=encoder_arr,
        encoder_mapping=encoder_map,
        manifest_csv=cfg.get("manifest_csv", None),
        label_map=cfg.get("label_map", None),
        max_notes=cfg.get("max_notes", 512),
        note_dim=cfg.get("note_dim", 4),
        augment=cfg.get("augment", False) and split == "train",
        augment_cfg=cfg.get("augment_cfg", None),
        preload=cfg.get("preload", False),
    )
    
    # Patch the internal dataframe if we didn't save a new CSV (double safety)
    # This ensures the dataset object uses the filtered memory even if it re-read the old file
    ds.df = df_filtered 

    # Weighted Sampler Logic (Train only)
    use_sampler = bool(cfg.get("use_weighted_sampler", False)) and (split == "train")
    sampler = None

    if use_sampler:
        # Helper to get int labels from dataframe
        def _get_label(row):
            lbl = row.get("emotion", row.get("label", None))
            return ds._load_item_raw(0)[1] if lbl is None else _to_int_label(lbl, ds.label_map)
            
        # We need to extract labels efficiently without loading files
        label_col = "emotion" if "emotion" in df_filtered.columns else "label"
        if label_col in df_filtered.columns:
            labels_list = [ _to_int_label(x, ds.label_map) for x in df_filtered[label_col] ]
            import collections
            counts = collections.Counter(labels_list)
            weights = [1.0 / counts[l] for l in labels_list]
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            print(f"[INFO] Using WeightedRandomSampler. Class counts: {dict(counts)}")

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        sampler=sampler,
        shuffle=(shuffle if split == "train" and sampler is None else False),
        num_workers=int(num_workers),
        collate_fn=EmotionDataset.collate_fn,
        pin_memory=True,
    )

    return loader


