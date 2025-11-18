# src/emotion_discriminator/ed_dataset.py
"""
Robust Dataset and dataloader utilities for Emotion Discriminator (Phase 1A).
This version is aligned to the project's directory tree and the config/ed_config.yaml
structure you provided.

Behavior & features
- Supports input_mode "latent" (preferred) and "notes".
- Automatically resolves encoder_feats for each split:
    * cfg may provide encoder_feats_path (global) or per-split:
        - train_encoder_feats_path, val_encoder_feats_path, test_encoder_feats_path
    * If not provided, tries common heuristics:
        - <dirname_of_split_csv>/encoder_feats.npy
        - data/splits/<split>/encoder_feats.npy
- Accepts encoder_feats.npy saved as:
    * regular numpy array (N x latent_dim) aligned to split CSV order
    * numpy object/dict mapping filename -> vector (common pattern)
- Uses a manifest CSV (cfg.manifest_csv) when resolving .npz paths for notes mode.
- Provides EmotionDataset and build_dataloader(cfg, split) with a sensible API.
- Includes lightweight augmentations for notes mode and optional preload.

Expected config keys (from config/ed_config.yaml):
- input_mode
- train_split_csv / val_split_csv / test_split_csv
- encoder_feats_path (optional) or per-split keys above + *_encoder_feats_path
- manifest_csv (optional)
- processed_dir
- max_notes, note_dim
- batch_size, augment, augment_cfg, preload
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
    # fallback to generic keys
    if "train_split_csv" in cfg and cfg["train_split_csv"] and split == "train":
        return cfg["train_split_csv"]
    raise ValueError(f"Missing split csv for '{split}' in config; expected key '{key}'.")


def _resolve_encoder_feats_for_split(cfg: Dict[str, Any], split: str) -> Optional[str]:
    # 1) per-split explicit key
    candidate_key = f"{split}_encoder_feats_path"
    if candidate_key in cfg and cfg[candidate_key]:
        return cfg[candidate_key]
    # 2) global key
    if "encoder_feats_path" in cfg and cfg["encoder_feats_path"]:
        return cfg["encoder_feats_path"]
    # 3) heuristic: look for encoder_feats.npy next to the split csv
    try:
        split_csv = _resolve_split_csv(cfg, split)
        csv_dir = os.path.dirname(split_csv) or "."
        cand = os.path.join(csv_dir, "encoder_feats.npy")
        if os.path.exists(cand):
            return cand
        # also check a common location: data/splits/<split>/encoder_feats.npy
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
        """
        Args:
            split_csv: CSV for the split (must contain a file identifier and emotion column)
            input_mode: 'latent' or 'notes'
            processed_dir: base directory to search for .npz files (notes mode)
            encoder_feats: N x D numpy array aligned to rows in split_csv (latent mode)
            encoder_mapping: dict filename->vector (latent mode)
            manifest_csv: optional CSV mapping file keys -> actual paths
            label_map: optional mapping str->int
        """
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

        # determine which column refers to the file key/path
        self.file_col = None
        for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
            if c in self.df.columns:
                self.file_col = c
                break
        if self.file_col is None:
            # if manifest exists and has a key column, prefer using it
            if self.manifest_csv and os.path.exists(self.manifest_csv):
                manifest_df = pd.read_csv(self.manifest_csv)
                # try to infer a common key between manifest and df
                common = set(self.df.columns).intersection(set(manifest_df.columns))
                if common:
                    self.file_col = list(common)[0]
            if self.file_col is None:
                raise ValueError("Split CSV must contain a file identifier column. One of: "
                                 "npz_path, file_key, full_path, filename, path, file")

        # load manifest mapping if present
        self._manifest_map = None
        if self.manifest_csv and os.path.exists(self.manifest_csv):
            try:
                mdf = pd.read_csv(self.manifest_csv)
                # expect columns like: file_key, relative_path or full_path
                # attempt to find a path-like column
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
                    # fallback: if the manifest has index mapping -> path
                    if "path" in mdf.columns:
                        # try to match by basename
                        self._manifest_map = {os.path.basename(p): p for p in mdf["path"].astype(str).tolist()}
            except Exception:
                warnings.warn("Failed to parse manifest_csv; continuing without manifest map.")

        # preload cache
        self._cached = [None] * len(self.df) if self.preload else None
        if self.preload:
            for i in range(len(self.df)):
                self._cached[i] = self._load_item_raw(i)

    def __len__(self):
        return len(self.df)

    def _resolve_npz_path(self, entry_val: str) -> str:
        if isinstance(entry_val, float) and np.isnan(entry_val):
            raise ValueError("Row has NaN in file path column.")
        v = str(entry_val)
        # if absolute path and exists, return
        if os.path.isabs(v) and os.path.exists(v):
            return v
        # if manifest map exists and has key
        if self._manifest_map and v in self._manifest_map:
            p = self._manifest_map[v]
            if os.path.isabs(p):
                if os.path.exists(p):
                    return p
            else:
                # join with project root
                if os.path.exists(p):
                    return p
                # join with processed_dir
                cand = os.path.join(self.processed_dir, p)
                if os.path.exists(cand):
                    return cand
        # try processed_dir join
        cand = os.path.join(self.processed_dir, v)
        if os.path.exists(cand):
            return cand
        if not v.endswith(".npz"):
            cand2 = cand + ".npz"
            if os.path.exists(cand2):
                return cand2
        # search for basename under processed_dir
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
            # Prefer mapping by file key if encoder_mapping exists
            if self.encoder_mapping:
                key = str(row[self.file_col])
                vec = self.encoder_mapping.get(key)
                if vec is None:
                    # also try basename
                    vec = self.encoder_mapping.get(os.path.basename(key))
                if vec is None:
                    raise KeyError(f"latent vector not found for key '{key}' in encoder_mapping.")
                latent = np.asarray(vec, dtype=np.float32)
                return latent, label_int, {"row": row.to_dict()}
            elif self.encoder_feats is not None:
                # attempt to use row-order alignment
                try:
                    latent = np.asarray(self.encoder_feats[idx], dtype=np.float32)
                    return latent, label_int, {"row": row.to_dict()}
                except Exception:
                    # fallback: if encoder_feats is dict-like saved as object array
                    if getattr(self.encoder_feats, "dtype", None) == np.object_:
                        try:
                            mapping = dict(self.encoder_feats.tolist())
                            key = str(row[self.file_col])
                            vec = mapping.get(key) or mapping.get(os.path.basename(key))
                            if vec is None:
                                raise KeyError
                            return np.asarray(vec, dtype=np.float32), label_int, {"row": row.to_dict()}
                        except Exception:
                            raise ValueError("encoder_feats appears object-dtype but couldn't map keys to rows.")
                    raise ValueError("encoder_feats does not index by row; provide encoder_mapping or a matching encoder_feats array.")
            else:
                raise FileNotFoundError("No encoder_feats provided for latent mode.")
        else:
            # notes mode: resolve .npz and load notes array
            entry = row[self.file_col]
            npz_path = self._resolve_npz_path(entry)
            data = np.load(npz_path, allow_pickle=True)
            if "notes" in data:
                notes = data["notes"]
            else:
                # try typical keys
                notes = data.get("arr_0", None)
                if notes is None:
                    # fallback: first file entry
                    first_key = list(data.files)[0]
                    notes = data[first_key]
            notes = np.asarray(notes, dtype=np.float32)
            # ensure shape (T, D)
            if notes.ndim == 1:
                try:
                    notes = notes.reshape(-1, self.note_dim)
                except Exception:
                    raise ValueError(f"Unable to reshape notes from {npz_path}")
            if notes.shape[1] != self.note_dim:
                # try transpose
                if notes.shape[0] == self.note_dim:
                    notes = notes.T
                else:
                    raise ValueError(f"Note dimension mismatch for {npz_path}. Expected second dim {self.note_dim}, got {notes.shape}")
            # pad / truncate to max_notes
            T = notes.shape[0]
            if T >= self.max_notes:
                notes = notes[: self.max_notes]
            else:
                pad = np.zeros((self.max_notes - T, self.note_dim), dtype=np.float32)
                notes = np.concatenate([notes, pad], axis=0)
            return notes, label_int, {"npz_path": npz_path, "row": row.to_dict()}

    def _augment_notes(self, notes: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        out = notes.copy()
        if cfg.get("noise_std", 0.0) > 0:
            std = float(cfg.get("noise_std", 0.01))
            for c in [1, 2, 3]:
                if c < out.shape[1]:
                    out[:, c] += np.random.normal(scale=std, size=(out.shape[0],))
        if cfg.get("dropout_prob", 0.0) > 0:
            p = float(cfg.get("dropout_prob", 0.05))
            mask = np.random.rand(out.shape[0]) >= p
            out[~mask] = 0.0
        if cfg.get("pitch_shift_prob", 0.0) > 0 and random.random() < cfg.get("pitch_shift_prob", 0.0):
            shift = random.choice([-1, 1])
            out[:, 0] += shift
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        if self.preload and self._cached is not None and self._cached[idx] is not None:
            raw, label_int, meta = self._cached[idx]
        else:
            raw, label_int, meta = self._load_item_raw(idx)

        if self.input_mode == "latent":
            tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        else:
            notes = raw
            if self.augment:
                notes = self._augment_notes(notes)
            tensor = torch.from_numpy(np.asarray(notes, dtype=np.float32))  # (T, D)

        return tensor, int(label_int), meta

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int, Dict[str, Any]]]) -> Dict[str, Any]:
        xs = [b[0] for b in batch]
        ys = torch.LongTensor([b[1] for b in batch])
        metas = [b[2] for b in batch]
        x = torch.stack(xs, dim=0)
        return {"x": x, "y": ys, "meta": metas}


def _load_encoder_feats_from_path(path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Load encoder_feats from a .npy file.
    Returns (array_or_none, mapping_or_none).
    If the npy contains an object-dtype mapping, returns mapping.
    If it contains a regular ndarray, returns ndarray.
    """
    if path is None:
        return None, None
    if not os.path.exists(path):
        return None, None
    arr = np.load(path, allow_pickle=True)
    if getattr(arr, "dtype", None) == np.object_:
        # try to coerce to dict
        try:
            mapping = dict(arr.tolist())
            return None, mapping
        except Exception:
            # maybe it's an array of arrays; return as-is
            try:
                arr2 = np.asarray(arr.tolist(), dtype=np.float32)
                return arr2, None
            except Exception:
                return None, None
    else:
        return arr, None

def build_dataloader(cfg: Dict[str, Any], split: str = "train", shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Build a DataLoader for the given split using configuration dict 'cfg'.

    - Auto-filters split CSV rows that lack latents (existing behavior).
    - Optionally uses a WeightedRandomSampler for the training split when
      cfg.get('use_weighted_sampler', False) is True.
    """
    from torch.utils.data import WeightedRandomSampler

    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of 'train', 'val', 'test'")

    split_csv = _resolve_split_csv(cfg, split)
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    # attempt to load encoder feats (array or mapping)
    encoder_path = _resolve_encoder_feats_for_split(cfg, split)
    encoder_arr, encoder_map = _load_encoder_feats_from_path(encoder_path) if encoder_path else (None, None)

    # load split CSV into dataframe (we will possibly filter it)
    df = pd.read_csv(split_csv)
    orig_len = len(df)

    # filtering logic (same as previous behavior)
    if cfg.get("input_mode", "latent") == "latent":
        if encoder_map:
            file_col = None
            for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
                if c in df.columns:
                    file_col = c
                    break
            if file_col is None:
                raise ValueError("Cannot find file identifier column to map against encoder_mapping.")
            def has_key(val):
                if pd.isna(val):
                    return False
                s = str(val)
                if s in encoder_map:
                    return True
                if os.path.basename(s) in encoder_map:
                    return True
                return False
            mask = df[file_col].apply(has_key)
            df_filtered = df[mask].reset_index(drop=True)
            dropped = orig_len - len(df_filtered)
            print(f"[ed_dataset] encoder_mapping present: dropped {dropped} rows from {orig_len} (kept {len(df_filtered)})")
        elif encoder_arr is not None:
            try:
                arr_len = int(getattr(encoder_arr, "shape", (len(encoder_arr),))[0])
            except Exception:
                arr_len = len(encoder_arr)
            if arr_len < orig_len:
                df_filtered = df.iloc[:arr_len].reset_index(drop=True)
                dropped = orig_len - arr_len
                print(f"[ed_dataset] encoder_feats ndarray shorter ({arr_len}) than CSV ({orig_len}): dropping last {dropped} rows.")
            else:
                df_filtered = df
                print(f"[ed_dataset] encoder_feats ndarray length OK ({arr_len}) for CSV ({orig_len}).")
        else:
            df_filtered = df
            print("[ed_dataset] Warning: input_mode=latent but no encoder_feats found; attempting to proceed (will likely error).")
    else:
        # notes mode: We must verify that the .npz files actually exist.
        print("[ed_dataset] Input mode is 'notes'. Verifying .npz file paths...")
        processed_dir = cfg.get('processed_dir', 'data/processed')

        # Find the file column
        file_col = None
        for c in ["npz_path", "file_key", "full_path", "filename", "path", "file"]:
            if c in df.columns:
                file_col = c
                break
        if file_col is None:
            raise ValueError("Cannot find file identifier column ('npz_path', 'filename', etc.) for 'notes' mode.")

        # This logic is adapted from your project's resolve_splits.py
        # to filter out missing .npz files before loading.
        keep_indices = []
        for idx, row in df.iterrows():
            fname = row[file_col]
            if pd.isna(fname):
                continue

            # Try a direct join first
            cand = os.path.join(processed_dir, fname if fname.endswith('.npz') else os.path.splitext(fname)[0] + '.npz')

            if os.path.exists(cand):
                keep_indices.append(idx)
                continue

            # Fallback: search processed dir for basename
            try:
                stem = os.path.splitext(os.path.basename(fname))[0]
                found = glob.glob(os.path.join(processed_dir, f"*{stem}*.npz"))
                if found and os.path.exists(found[0]):
                    keep_indices.append(idx)
                else:
                    print(f"[ed_dataset] Warning: processed file for {fname} not found. Skipping.")
            except Exception:
                print(f"[ed_dataset] Warning: Could not parse file path {fname}. Skipping.")

        df_filtered = df.loc[keep_indices].reset_index(drop=True)
        dropped = orig_len - len(df_filtered)
        if dropped > 0:
            print(f"[ed_dataset] 'notes' mode: dropped {dropped} rows from {orig_len} (kept {len(df_filtered)}) due to missing .npz files.")

    # if filtering happened, write filtered csv to a safe location and use that path
    filtered_csv_path = split_csv
    if len(df_filtered) != orig_len:
        split_dir = os.path.dirname(split_csv) or os.path.join("data", "splits", split)
        os.makedirs(split_dir, exist_ok=True)
        base = os.path.basename(split_csv)
        filtered_csv_path = os.path.join(split_dir, f"auto_filtered_{base}")
        df_filtered.to_csv(filtered_csv_path, index=False)
        print(f"[ed_dataset] Wrote filtered split CSV to: {filtered_csv_path}")

    # build the dataset using the filtered CSV
    manifest_csv = cfg.get("manifest_csv", None)
    ds = EmotionDataset(
        split_csv=filtered_csv_path,
        input_mode=cfg.get("input_mode", "latent"),
        processed_dir=cfg.get("processed_dir", "data/processed"),
        encoder_feats=encoder_arr,
        encoder_mapping=encoder_map,
        manifest_csv=manifest_csv,
        label_map=cfg.get("label_map", None),
        max_notes=cfg.get("max_notes", 512),
        note_dim=cfg.get("note_dim", 4),
        augment=cfg.get("augment", False) and split == "train",
        augment_cfg=cfg.get("augment_cfg", None),
        preload=cfg.get("preload", False),
    )

    # decide whether to use WeightedRandomSampler (train only)
    use_sampler = bool(cfg.get("use_weighted_sampler", False)) and (split == "train")
    sampler = None

    if use_sampler:
        # Compute integer labels from the filtered dataframe (fast, no iteration over dataset)
        # We rely on the same label encoding logic as _to_int_label
        def _to_int_local(lbl):
            if isinstance(lbl, (int, np.integer)):
                return int(lbl)
            s = str(lbl).strip().lower()
            lm = cfg.get("label_map", None)
            if isinstance(lm, dict) and s in lm:
                return int(lm[s])
            if s in DEFAULT_LABELS:
                return DEFAULT_LABELS.index(s)
            try:
                return int(float(s))
            except Exception:
                raise ValueError(f"Could not convert label '{lbl}' to int. Provide a label_map.")

        # extract labels from df_filtered; prefer column 'emotion' or 'label'
        label_col = "emotion" if "emotion" in df_filtered.columns else ("label" if "label" in df_filtered.columns else None)
        if label_col is None:
            raise ValueError("Cannot find 'emotion' or 'label' column in split CSV for weighted sampler.")
        labels_list = df_filtered[label_col].apply(_to_int_local).tolist()

        # compute class counts
        import collections
        counts = collections.Counter(labels_list)
        num_classes = max(counts.keys()) + 1 if counts else 0
        # sample weight = 1 / class_count[label]
        sample_weights = [1.0 / counts[int(l)] for l in labels_list]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[ed_dataset] Using WeightedRandomSampler: classes={dict(counts)}, samples={len(sample_weights)}")

    # build dataloader: use sampler if provided (disable shuffle)
    if sampler is not None:
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 64)),
            sampler=sampler,
            num_workers=int(num_workers),
            collate_fn=EmotionDataset.collate_fn,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=int(cfg.get("batch_size", 64)),
            shuffle=shuffle if split == "train" else False,
            num_workers=int(num_workers),
            collate_fn=EmotionDataset.collate_fn,
            pin_memory=True,
        )

    return loader


if __name__ == "__main__":
    # quick smoke test - won't run without data, but verifies API
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ed_config.yaml")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    loader = build_dataloader(cfg, split=args.split, shuffle=False, num_workers=0)
    print("Dataloader created. Length:", len(loader.dataset))
    try:
        batch = next(iter(loader))
        print("Batch x shape:", batch["x"].shape, "y shape:", batch["y"].shape)
    except Exception as e:
        print("Smoke test failed to load batch (expected if no local data):", e)