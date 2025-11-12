#!/usr/bin/env python3
"""
Encoder inference utility (robust manifest handling).

Usage example:
  python -m src.ae.encode \
    --encoder data/models/ae/encoder_final.pth \
    --manifest data/splits/train_split.csv \
    --out_file data/splits/train/encoder_feats.npy \
    --processed_dir data/processed \
    --config config/ae_config.yaml
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure repo root on path when run as module
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Absolute imports from project
from src.ae.path_utils import load_config, ensure_dir
# Import ConvEncoder definition from the ae training module
from .model import ConvEncoder

def resolve_processed_path(processed_dir, filename):
    """Try direct .npz match, then stem search in processed_dir."""
    if not isinstance(filename, str):
        return None
    # direct .npz path
    if filename.lower().endswith('.npz'):
        p = filename if os.path.isabs(filename) else os.path.join(processed_dir, filename)
        if os.path.exists(p):
            return p
    # if filename is full path and exists
    if os.path.exists(filename):
        # if it's .npz return it
        if filename.lower().endswith('.npz'):
            return filename
        # else search by stem
        stem = os.path.splitext(os.path.basename(filename))[0]
    else:
        stem = os.path.splitext(os.path.basename(filename))[0]
    # search processed_dir by stem
    candidates = glob.glob(os.path.join(processed_dir, f"*{stem}*.npz"))
    if candidates:
        return candidates[0]
    return None

def build_file_list_from_manifest(manifest_df, processed_dir):
    preferred_cols = ['npz_filename', 'processed_file', 'processed', 'full_path', 'filepath', 'file', 'filename', 'file_key']
    col_found = None
    for c in preferred_cols:
        if c in manifest_df.columns:
            col_found = c
            break
    if col_found is None:
        raise KeyError(f"Manifest must contain one of {preferred_cols}. Found columns: {list(manifest_df.columns)}")

    files = []
    file_names = []
    for _, r in manifest_df.iterrows():
        raw_cell = r[col_found]
        # Try direct resolve
        p = resolve_processed_path(processed_dir, str(raw_cell))
        if p is not None:
            files.append(p); file_names.append(os.path.basename(p)); continue
        # Fallback: if there is explicit npz_filename column, try it
        if 'npz_filename' in manifest_df.columns:
            npzcell = r['npz_filename']
            p2 = resolve_processed_path(processed_dir, str(npzcell))
            if p2 is not None:
                files.append(p2); file_names.append(os.path.basename(p2)); continue
        # Could not resolve
        print("Warning: could not locate processed .npz for manifest row:", dict(r))
    return files, file_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder_final.pth")
    parser.add_argument("--manifest", type=str, required=True, help="CSV listing files (various column names supported)")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed .npz dir")
    parser.add_argument("--out_file", type=str, required=True, help="Output .npy file for latents")
    parser.add_argument("--config", type=str, default="config/ae_config.yaml", help="Config with LATENT_DIM, MAX_NOTES")
    args = parser.parse_args()

    cfg = load_config(args.config)
    latent_dim = cfg['LATENT_DIM']
    processed_dir = args.processed_dir

    manifest = pd.read_csv(args.manifest)
    files, file_names = build_file_list_from_manifest(manifest, processed_dir)
    print("Found", len(files), "processed files to encode (processed_dir=", processed_dir, ")")

    if len(files) == 0:
        raise RuntimeError("No processed files found. Check manifest and processed_dir paths.")

    # build encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ConvEncoder(in_channels=4, latent_dim=cfg['LATENT_DIM'])
    # warm-build linear by passing dummy
    dummy = torch.zeros(1, cfg['MAX_NOTES'], 4)
    with torch.no_grad():
        _ = encoder(dummy)
    # load weights
    state = torch.load(args.encoder, map_location=device)
    # state may be a dict containing 'model_state' or a raw state_dict
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    # state is expected to be encoder state dict (not full ae)
    try:
        encoder.load_state_dict(state)
    except Exception:
        # try if state itself is encoder-only
        try:
            encoder.load_state_dict(state)
        except Exception as e:
            raise RuntimeError("Failed to load encoder state. Check encoder checkpoint format.") from e

    encoder.to(device)
    encoder.eval()

    latents = []
    for p in tqdm(files, desc="Encoding"):
        data = np.load(p, allow_pickle=True)
        if 'notes' not in data:
            print("Warning: 'notes' missing in", p); continue
        notes = data['notes'].astype(np.float32)
        notes_tensor = torch.from_numpy(notes).unsqueeze(0).to(device)  # (1, MAX_NOTES, 4)
        with torch.no_grad():
            z = encoder(notes_tensor)
            z = z.cpu().numpy()[0]
        latents.append(z)

    latents = np.stack(latents, axis=0)
    ensure_dir(os.path.dirname(args.out_file) or ".")
    np.save(args.out_file, latents)
    # save mapping manifest
    map_df = pd.DataFrame({'processed_file': file_names, 'latent_index': np.arange(len(file_names))})
    map_df.to_csv(os.path.splitext(args.out_file)[0] + "_manifest.csv", index=False)
    print("Saved latents ->", args.out_file)
    print("Saved manifest ->", os.path.splitext(args.out_file)[0] + "_manifest.csv")

if __name__ == "__main__":
    main()
