#!/usr/bin/env python3
"""
VAE Encoder inference utility.

Loads the trained VAE model and extracts the 'mu' (mean)
of the latent distribution as the representative latent vector.

Usage example:
  python -m src.ae.encode \
    --model data/models/ae/ae_best.pth \
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
from torch.utils.data import DataLoader


# Ensure repo root on path when run as module
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

# Absolute imports from project
from src.ae.path_utils import load_config, ensure_dir
from src.ae.model import  VAE # Import the VAE class
from src.ae.dataset import MIDIDataset # Use the same dataset for consistency
def resolve_paths_for_encode(df, processed_dir):
        files = []
        for _, row in df.iterrows():
            fname = row['npz_path']
            candidates = [
                os.path.join(processed_dir, fname if fname.endswith('.npz') else os.path.splitext(fname)[0] + '.npz')
            ]
            # Fallback: search processed dir for any containing filename stem
            if not os.path.exists(candidates[0]):
                stem = os.path.splitext(fname)[0]
                found = glob.glob(os.path.join(processed_dir, f"*{stem}*.npz"))
                if found:
                    candidates = [found[0]]
            
            if os.path.exists(candidates[0]):
                files.append(candidates[0])
            else:
                # This will now match the training script's behavior
                print(f"Warning: processed file for {fname} not found in {processed_dir}. Skipping.")
        return files
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to full ae_best.pth VAE model")
    parser.add_argument("--manifest", type=str, required=True, help="CSV listing files")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Processed .npz dir")
    parser.add_argument("--out_file", type=str, required=True, help="Output .npy file for latents (mu)")
    parser.add_argument("--config", type=str, default="config/ae_config.yaml", help="Config with LATENT_DIM, MAX_NOTES")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # --- Resolve paths from config ---
    if not os.path.isabs(args.manifest):
        args.manifest = os.path.join(REPO_ROOT, args.manifest)
    if not os.path.isabs(args.processed_dir):
        args.processed_dir = os.path.join(REPO_ROOT, args.processed_dir)
    if not os.path.isabs(args.out_file):
        args.out_file = os.path.join(REPO_ROOT, args.out_file)
    if not os.path.isabs(args.model):
        args.model = os.path.join(REPO_ROOT, args.model)
        
    cfg['PROCESSED_DIR'] = args.processed_dir
    cfg['SPLITS_DIR'] = os.path.dirname(args.manifest)
    split_name = Path(args.manifest).stem.replace('_split', '')

    # --- Use the same MIDIDataset from training ---
    try:
        
        manifest_df = pd.read_csv(args.manifest)
        manifest_paths = resolve_paths_for_encode(manifest_df, args.processed_dir)
        
    except KeyError:
        print(f"Error: 'npz_path' column not found in {args.manifest}.")
        print("Please run create_splits.py first.")
        sys.exit(1)
        
    dataset = MIDIDataset(
        manifest_paths,
        cfg,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Found {len(dataset)} files to encode for split: {split_name}")

    if len(dataset) == 0:
        raise RuntimeError("No processed files found. Check manifest and processed_dir paths.")

    # --- Build VAE model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(cfg).to(device)
    print("Initializing dynamic model layers...")
    try:
        dummy_notes = torch.zeros(1, cfg['MAX_NOTES'], 4).to(device)
        with torch.no_grad():
            _ = model.encoder(dummy_notes)
    except Exception as e:
        print(f"Error during model initialization with dummy batch: {e}")
        print("Check your config's MAX_NOTES setting.")
        sys.exit(1)
        
    # Load the VAE state dict
    print(f"Loading model from {args.model}")
    ckpt = torch.load(args.model, map_location=device)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    latents = []
    with torch.no_grad():
        for batch_notes, _ in tqdm(loader, desc=f"Encoding {split_name} split"):
            batch_notes = batch_notes.to(device)
            
            # --- VAE forward pass ---
            # We only care about 'mu', the mean of the latent space.
            # This is the most stable vector to use for classification.
            recon, z, mu, log_var = model(batch_notes)
            
            latents.append(mu.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    
    ensure_dir(os.path.dirname(args.out_file))
    np.save(args.out_file, latents)
    
    print(f"Saved latents ({latents.shape}) -> {args.out_file}")

if __name__ == "__main__":
    main()