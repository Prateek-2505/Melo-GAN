#!/usr/bin/env python3
"""
DIAGNOSIS TOOL
Checks data scaling, variance, and feature validity.
"""

import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.gan.dataset import GANDataset

def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

def diagnose(config_path):
    print("="*40)
    print("      SYSTEM DIAGNOSIS REPORT")
    print("="*40)
    
    cfg = load_config(config_path)
    
    # 1. Check Paths
    print(f"[CHECK] Split CSV: {cfg['TRAIN_SPLIT']}")
    if not os.path.exists(cfg['TRAIN_SPLIT']):
        print(f"   [FAIL] File not found!")
        return

    # 2. Load Dataset (First 100 samples)
    print("[CHECK] Loading Training Data...")
    try:
        ds = GANDataset(
            cfg['TRAIN_SPLIT'], 
            processed_dir=cfg['PROCESSED_DIR'],
            notes_npy=cfg.get('ENCODER_FEATS_TRAIN', '').replace('encoder_feats', 'notes'), # Guessing path
            emotion_npy=cfg.get('ENCODER_FEATS_TRAIN', '').replace('encoder_feats', 'emotion'),
            numeric_features_npy=cfg.get('ENCODER_FEATS_TRAIN', '').replace('encoder_feats', 'numeric_features'),
            latent_feats=np.load(cfg['ENCODER_FEATS_TRAIN']) if os.path.exists(cfg['ENCODER_FEATS_TRAIN']) else None
        )
    except Exception as e:
        print(f"   [FAIL] Dataset load error: {e}")
        return

    loader = DataLoader(ds, batch_size=32, shuffle=True)
    try:
        notes, emotions, latents, numeric = next(iter(loader))
    except StopIteration:
        print("   [FAIL] Dataset is empty!")
        return

    # 3. Analyze Real Data Scale (CRITICAL)
    print("-" * 20)
    print("DATA SCALING CHECKS")
    print(f"Notes Shape: {notes.shape} (Batch, Time, Dim)")
    print(f"   Pitch range:    {notes[:,:,0].min():.2f} to {notes[:,:,0].max():.2f}")
    print(f"   Velocity range: {notes[:,:,1].min():.2f} to {notes[:,:,1].max():.2f}")
    print(f"   Duration range: {notes[:,:,2].min():.2f} to {notes[:,:,2].max():.2f}")
    
    # Heuristic Check
    if notes[:,:,0].max() > 2.0:
        print("   [WARNING] Real data appears UN-NORMALIZED (Values > 1.0).")
        print("             GANs struggle to generate raw MIDI values (0-127).")
        print("             Data should be scaled to [-1, 1] or [0, 1].")
    else:
        print("   [OK] Real data appears normalized.")

    # 4. Analyze Numeric Features (Conditioning)
    print("-" * 20)
    print("NUMERIC FEATURE CHECKS")
    print(f"Numeric Shape: {numeric.shape}")
    print(f"   Min: {numeric.min():.2f}, Max: {numeric.max():.2f}")
    print(f"   Mean: {numeric.mean():.2f}, Std Dev: {numeric.std():.2f}")
    
    if numeric.std() < 0.01:
        print("   [CRITICAL FAIL] Numeric features have almost ZERO variance.")
        print("                   The model cannot learn 'Happy' vs 'Sad' if the inputs are identical!")
    else:
        print("   [OK] Numeric features show variance.")

    # 5. Analyze Latents (AE Output)
    print("-" * 20)
    print("LATENT VECTOR CHECKS")
    if latents is not None:
        print(f"Latent Shape: {latents.shape}")
        print(f"   Min: {latents.min():.2f}, Max: {latents.max():.2f}")
        print(f"   Std Dev: {latents.std():.2f}")
        if latents.std() < 0.1:
            print("   [WARNING] Latent vectors are very collapsed (low variance).")
            print("             Your Autoencoder might have suffered Posterior Collapse.")
    else:
        print("   [FAIL] No latent vectors found (None).")

if __name__ == "__main__":
    diagnose("config/gan_config.yaml")
