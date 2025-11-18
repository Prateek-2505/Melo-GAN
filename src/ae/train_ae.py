#!/usr/bin/env python3
"""
VAE training script.
Replaces the standard AE to fix posterior collapse.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# --- Local Imports ---
from .path_utils import load_config, ensure_dir
from .dataset import MIDIDataset
from .model import VAE  # --- UPDATED: Import VAE ---
from .midi_utils import save_recon_midi
from .resolve_splits import resolve

# -------------- Loss Function -----------------------

def vae_loss(recon, target, mu, log_var, beta):
    """
    Calculates the VAE loss.
    Loss = Reconstruction_Loss + Beta * KL_Divergence_Loss
    """
    # 1. Reconstruction Loss (MSE)
    recon_loss = F.mse_loss(recon, target)
    
    # 2. KL Divergence Loss
    #    Measures how far the latent space is from a standard N(0, 1) distribution
    #    Forces variance (prevents collapse)
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # 3. Total Loss
    total_loss = recon_loss + beta * kld_loss
    
    return total_loss, recon_loss, kld_loss

# -------------- Training loop -----------------------

def train(cfg):
    model_dir = cfg.get('CHECKPOINT_DIR', 'models/ae')
    log_dir = cfg.get('LOG_DIR', 'experiments/ae')
    
    train_files, val_files = resolve(cfg)

    print(f"Train files: {len(train_files)}   Val files: {len(val_files)}")

    train_ds = MIDIDataset(train_files, cfg, augment=True)
    val_ds = MIDIDataset(val_files, cfg, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=2, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- UPDATED: Use VAE ---
    model = VAE(cfg).to(device)
    
    # Build linear layers by running a dummy batch
    dummy = torch.zeros(1, cfg['MAX_NOTES'], 4).to(device)
    with torch.no_grad():
        _ = model.encoder(dummy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('LR', 0.0001)), weight_decay=float(cfg.get('WEIGHT_DECAY', 0.00001)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-6)
    
    writer = SummaryWriter(log_dir)

    best_val = float('inf')
    patience = cfg.get('EARLY_STOP_PATIENCE', 10)
    no_improve = 0
    
    # --- VAE KL Annealing ---
    # We slowly increase the KLD loss weight (beta) from 0 to 1.
    # This lets the model learn reconstruction first, then forces variance.
    kld_warmup_epochs = cfg.get('KLD_WARMUP_EPOCHS', 25)
    final_beta = float(cfg.get('BETA', 1.0))

    fixed_val_paths = val_files[:min(6, len(val_files))]

    for epoch in range(1, cfg['EPOCHS'] + 1):
        model.train()
        
        train_loss_total, train_loss_recon, train_loss_kld = 0.0, 0.0, 0.0
        n_batches = 0
        
        # Calculate beta for this epoch
        # CHANGE IT BACK TO THIS
        # beta = min(1.0, epoch / kld_warmup_epochs)
        beta = min(final_beta, (epoch / kld_warmup_epochs) * final_beta)
        if epoch >= kld_warmup_epochs:
            beta = final_beta
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train [B=0.00]")
        for batch_notes, _ in pbar:
            batch_notes = batch_notes.to(device)
            
            # --- UPDATED: VAE forward pass ---
            recon, z, mu, log_var = model(batch_notes)
            
            # --- UPDATED: VAE loss ---
            loss, recon_loss, kld_loss = vae_loss(recon, batch_notes, mu, log_var, beta)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            train_loss_recon += recon_loss.item()
            train_loss_kld += kld_loss.item()
            n_batches += 1
            
            pbar.set_description(f"Epoch {epoch} train [B={beta:.2f}]")
            
        train_loss_total /= max(1, n_batches)
        train_loss_recon /= max(1, n_batches)
        train_loss_kld /= max(1, n_batches)

        # validation
        model.eval()
        val_loss_total, val_loss_recon, val_loss_kld = 0.0, 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_notes, fnames in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                batch_notes = batch_notes.to(device)
                
                # --- UPDATED: VAE forward pass ---
                recon, z, mu, log_var = model(batch_notes)
                
                # --- UPDATED: VAE loss (full beta for val) ---
                loss, recon_loss, kld_loss = vae_loss(recon, batch_notes, mu, log_var, beta=1.0)
                
                val_loss_total += loss.item()
                val_loss_recon += recon_loss.item()
                val_loss_kld += kld_loss.item()
                val_batches += 1
                
        val_loss_total /= max(1, val_batches)
        val_loss_recon /= max(1, val_batches)
        val_loss_kld /= max(1, val_batches)

        scheduler.step(val_loss_total)

        print(f"[Epoch {epoch}] Train: {train_loss_total:.6f} (Recon: {train_loss_recon:.6f}, KLD: {train_loss_kld:.6f}) | "
              f"Val: {val_loss_total:.6f} (Recon: {val_loss_recon:.6f}, KLD: {val_loss_kld:.6f})")
        
        writer.add_scalar("loss/train_total", train_loss_total, epoch)
        writer.add_scalar("loss/train_recon", train_loss_recon, epoch)
        writer.add_scalar("loss/train_kld", train_loss_kld, epoch)
        writer.add_scalar("loss/val_total", val_loss_total, epoch)
        writer.add_scalar("loss/val_recon", val_loss_recon, epoch)
        writer.add_scalar("loss/val_kld", val_loss_kld, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("beta", beta, epoch)

        # save reconstructions (this code is fine)
        recon_dir = cfg.get('RECON_DIR', os.path.join(log_dir, 'reconstructions'))
        ensure_dir(recon_dir)
        if epoch % cfg.get('RECON_FREQ', 1) == 0:
            for p in fixed_val_paths:
                try:
                    data = np.load(p, allow_pickle=True)
                    notes_in = data['notes'].astype(np.float32)
                    notes_tensor = torch.from_numpy(notes_in).unsqueeze(0).to(device)
                    with torch.no_grad():
                        recon, _, _, _ = model(notes_tensor) # VAE returns 4 items
                    recon_np = recon.detach().cpu().numpy()[0]
                    base = os.path.splitext(os.path.basename(p))[0]
                    prefix = f"ep{epoch}_{base}"
                    save_recon_midi(notes_in, recon_np, recon_dir, prefix) # Removed tempo, as it's not loaded
                except Exception as e:
                    print("recon save failed for", p, ":", e)

        # save checkpoint if improved (track total val loss)
        if val_loss_total < best_val:
            best_val = val_loss_total
            best_path = os.path.join(model_dir, "ae_best.pth")
            # --- UPDATED: Save the full VAE state, not just encoder ---
            # We save the full model. The 'encode.py' script will need to be
            # updated to load this and extract the encoder part.
            torch.save({'epoch': epoch, 'model_state': model.state_dict()}, best_path)
            no_improve = 0
            print("Saved new best model ->", best_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print("No improvement for", patience, "epochs. Early stopping.")
            break

    writer.close()
    print("Training complete. Best val:", best_val)
    
    # final save of the full model
    final_model_path = os.path.join(model_dir, "ae_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Saved final model:", final_model_path)
    
    # --- NOTE ---
    # We save the *full VAE* state. The `encode.py` script will need to be
    # updated to load the VAE and use its `encoder`, `fc_mu`, and `fc_log_var`
    # parts to generate the final latents.

# -------------- CLI -----------------------
def main():
    # ... (This main() function is fine, no changes needed) ...
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ae_config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    config_path = args.config
    if not os.path.exists(config_path):
        alt = os.path.join(repo_root, config_path)
        if os.path.exists(alt):
            config_path = alt
        else:
            alt2 = os.path.join(repo_root, "config", os.path.basename(config_path))
            if os.path.exists(alt2):
                config_path = alt2

    from src.ae.path_utils import load_config, ensure_dir
    cfg = load_config(config_path)

    if not os.path.isabs(cfg.get('CHECKPOINT_DIR','')):
        cfg['CHECKPOINT_DIR'] = os.path.join(repo_root, cfg.get('CHECKPOINT_DIR','models/ae'))
    if not os.path.isabs(cfg.get('LOG_DIR','')):
        cfg['LOG_DIR'] = os.path.join(repo_root, cfg.get('LOG_DIR','experiments/ae'))

    ensure_dir(cfg.get('CHECKPOINT_DIR', 'models/ae'))
    ensure_dir(cfg.get('LOG_DIR', 'experiments/ae'))
    train(cfg)


if __name__ == "__main__":
    main()