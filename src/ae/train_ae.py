#!/usr/bin/env python3
"""
Autoencoder training script for MELO-GAN (Pipeline B).

Usage:
    python src/ae/train_ae.py --config configs/ae_config.yaml
"""

import os
import sys
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# --- Local Imports ---
from .path_utils import load_config, ensure_dir
from .dataset import MIDIDataset
from .model import Autoencoder
from .midi_utils import save_recon_midi
from .resolve_splits import resolve
# -------------- Training loop -----------------------

def train(cfg):
    model_dir = cfg.get('CHECKPOINT_DIR', 'models/ae')
    log_dir = cfg.get('LOG_DIR', 'experiments/ae')
    
    train_files,val_files = resolve(cfg)

    print(f"Train files: {len(train_files)}   Val files: {len(val_files)}")

    # datasets & loaders
    train_ds = MIDIDataset(train_files, cfg, augment=True)
    val_ds = MIDIDataset(val_files, cfg, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=2, drop_last=False)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(cfg).to(device)
    
    

    # If encoder linear wasn't built, we ensure it's built by running a dummy through the encoder
    dummy = torch.zeros(1, cfg['MAX_NOTES'], 4).to(device)
    with torch.no_grad():
        _ = model.encoder(dummy)
    # Ensure encoder's linear layer is on the correct device (was a bug in original)
    if model.encoder._linear is not None:
        model.encoder._linear = model.encoder._linear.to(device)


    # optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(cfg.get('LR', 1e-3)),weight_decay=float(cfg.get('WEIGHT_DECAY', 1e-5)))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir)

    best_val = float('inf')
    patience = cfg.get('EARLY_STOP_PATIENCE', 10)
    no_improve = 0

    # pre-define fixed val examples for recon saving
    fixed_val_paths = val_files[:min(6, len(val_files))]

    for epoch in range(1, cfg['EPOCHS'] + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_notes, _ in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            batch_notes = batch_notes.to(device)  # (B, MAX_NOTES, 4)
            recon, z = model(batch_notes)
            loss = criterion(recon, batch_notes)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        # validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_notes, fnames in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                batch_notes = batch_notes.to(device)
                recon, z = model(batch_notes)
                loss = criterion(recon, batch_notes)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(1, val_batches)

        # scheduler step
        scheduler.step(val_loss)

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # save reconstructions for fixed examples
        # save reconstructions for fixed examples (safe, no-grad + detach)
        recon_dir = cfg.get('RECON_DIR', os.path.join(log_dir, 'reconstructions'))
        ensure_dir(recon_dir)
        if epoch % cfg.get('RECON_FREQ', 1) == 0:
            for p in fixed_val_paths:
                try:
                    data = np.load(p, allow_pickle=True)
                    notes_in = data['notes'].astype(np.float32)
                    notes_tensor = torch.from_numpy(notes_in).unsqueeze(0).to(device)
                    # DO inference without tracking grads and detach before numpy
                    with torch.no_grad():
                        recon, _ = model(notes_tensor)
                    recon_np = recon.detach().cpu().numpy()[0]
                    # write mids
                    base = os.path.splitext(os.path.basename(p))[0]
                    prefix = f"ep{epoch}_{base}"
                    save_recon_midi(notes_in, recon_np, recon_dir, prefix, tempo=float(data.get('tempo', 120.0)))
                except Exception as e:
                    print("recon save failed for", p, ":", e)


        # save checkpoint if improved
        ckpt_path = os.path.join(model_dir, f"ae_epoch{epoch}_val{val_loss:.6f}.pth")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)
        # keep best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(model_dir, f"ae_best.pth")
            torch.save({'epoch': epoch, 'model_state': model.encoder.state_dict()}, best_path)  # save encoder only for GAN init
            no_improve = 0
            print("Saved new best encoder ->", best_path)
        else:
            no_improve += 1

        # early stop
        if no_improve >= patience:
            print("No improvement for", patience, "epochs. Early stopping.")
            break

    writer.close()
    print("Training complete. Best val:", best_val)
    # final save of encoder
    final_encoder = os.path.join(model_dir, "encoder_final.pth")
    torch.save(model.encoder.state_dict(), final_encoder)
    print("Saved final encoder:", final_encoder)

# -------------- CLI -----------------------
def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ae_config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    # Ensure we know the script directory (src/ae)
    script_dir = os.path.dirname(os.path.realpath(__file__))       # e.g. /path/to/project/src/ae
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))   # project root

    # Resolve config path robustly:
    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to repo root
        alt = os.path.join(repo_root, config_path)
        if os.path.exists(alt):
            config_path = alt
        else:
            # try the sibling 'config' directory (your layout)
            alt2 = os.path.join(repo_root, "config", os.path.basename(config_path))
            if os.path.exists(alt2):
                config_path = alt2

    # Now load config via path_utils (absolute import)
    from src.ae.path_utils import load_config, ensure_dir
    cfg = load_config(config_path)

    # ensure output dirs use your project 'data' layout if present
    # normalize checkpoint/log dirs if they are relative
    if not os.path.isabs(cfg.get('CHECKPOINT_DIR','')):
        cfg['CHECKPOINT_DIR'] = os.path.join(repo_root, cfg.get('CHECKPOINT_DIR','models/ae'))
    if not os.path.isabs(cfg.get('LOG_DIR','')):
        cfg['LOG_DIR'] = os.path.join(repo_root, cfg.get('LOG_DIR','experiments/ae'))

    ensure_dir(cfg.get('CHECKPOINT_DIR', 'models/ae'))
    ensure_dir(cfg.get('LOG_DIR', 'experiments/ae'))
    train(cfg)


if __name__ == "__main__":
    main()