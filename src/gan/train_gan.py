#!/usr/bin/env python3
"""
Train MELO-GAN with WGAN-GP, Numeric Feature Conditioning, and Pre-trained Emotion Discriminator.

Architecture:
- Generator (G): Conditioned on Noise + Latent + Numeric Features.
- Discriminator (D): WGAN-GP Critic (Real vs Fake), conditioned on Numeric Features.
- Emotion Discriminator (ED): Pre-trained Classifier (Fixed/Frozen), judges emotion of generated notes.
"""

import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# --- UPDATED IMPORTS ---
from src.gan.models import Generator, Discriminator  # D is the WGAN-GP Critic
from src.emotion_discriminator.ed_model import EmotionDiscriminator # ED from its own file
from src.gan.feature_encoder import FeatureEncoder
from src.gan.dataset import GANDataset
from src.gan.utils import (
    seed_everything, 
    weights_init, 
    load_ae_decoder_into_generator, 
    emotion_to_index,
    compute_gradient_penalty
)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def prepare_dataset(cfg, split_csv, latent_feats_path=None):
    # (This function is correct, no changes needed)
    splits_dir = cfg.get('SPLITS_DIR', 'data/splits')
    if not os.path.isabs(split_csv) and not os.path.exists(split_csv):
        candidate = os.path.join(splits_dir, split_csv)
        if os.path.exists(candidate): split_csv = candidate
    split_name = Path(split_csv).stem
    notes_npy = os.path.join(splits_dir, split_name, "notes.npy")
    emo_npy = os.path.join(splits_dir, split_name, "emotion.npy")
    numeric_npy = os.path.join(splits_dir, split_name, "numeric_features.npy")
    latent_feats = None
    if latent_feats_path and os.path.exists(latent_feats_path):
        latent_feats = np.load(latent_feats_path)
    return GANDataset(
        split_csv, 
        processed_dir=cfg.get('PROCESSED_DIR', 'data/processed'),
        notes_npy=notes_npy if os.path.exists(notes_npy) else None,
        emotion_npy=emo_npy if os.path.exists(emo_npy) else None,
        latent_feats=latent_feats,
        numeric_features_npy=numeric_npy if os.path.exists(numeric_npy) else None,
        numeric_input_dim=cfg.get('NUMERIC_INPUT_DIM', 6),
        latent_dim=cfg['LATENT_DIM']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gan_config.yaml", help="Path to the main GAN config")
    parser.add_argument("--ed_config", type=str, default="config/ed_config.yaml", help="Path to the ED config")
    parser.add_argument("--ed_ckpt", type=str, default="data/models/ed/ed_best.pth")
    args = parser.parse_args()
    
    # Load BOTH config files
    cfg = load_config(args.config)       # Main GAN config
    ed_cfg = load_config(args.ed_config) # Emotion Discriminator config

    seed_everything(cfg.get("SEED", 42))
    device = torch.device(cfg.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using main device: {device}")

    # --- 1. Prepare Data ---
    train_ds = prepare_dataset(cfg, cfg['TRAIN_SPLIT'], cfg.get('ENCODER_FEATS_TRAIN', None))
    train_loader = DataLoader(train_ds, batch_size=cfg.get('BATCH_SIZE', 32), shuffle=True, num_workers=4, drop_last=True)
    print(f"Train set size: {len(train_ds)}")

    # --- 2. Instantiate Models ---
    
    # A. Numeric Encoder
    numeric_input_dim = cfg.get('NUMERIC_INPUT_DIM', 6)
    numeric_embed_dim = cfg.get('ENCODER_OUT_DIM', 128)
    E_num = FeatureEncoder(
        in_dim=numeric_input_dim,
        hidden_dims=cfg.get('ENCODER_HIDDEN', [256, 128]),
        out_dim=numeric_embed_dim
    ).to(device)
    
    # B. Generator
    G = Generator(
        noise_dim=cfg['NOISE_DIM'], 
        latent_dim=cfg['LATENT_DIM'], 
        mode=cfg.get('INTEGRATION_MODE', 'conditioning'), 
        max_notes=cfg['MAX_NOTES'], 
        note_dim=cfg['NOTE_DIM'],
        numeric_embed_dim=numeric_embed_dim
    ).to(device)
    
    # C. Discriminator (WGAN)
    D_discriminator = Discriminator(
        max_notes=cfg['MAX_NOTES'], 
        note_dim=cfg['NOTE_DIM'],
        numeric_embed_dim=numeric_embed_dim
    ).to(device)
    
    # D. Emotion Discriminator (Classifier)
    print("[INFO] Instantiating Emotion Discriminator using ed_config.yaml")
    D_emotion = EmotionDiscriminator(ed_cfg).to(device)

    # --- 3. Initialization & Loading ---
    E_num.apply(weights_init)
    G.apply(weights_init)
    D_discriminator.apply(weights_init)
    
    # Load Pre-trained Emotion Discriminator
    ed_path = args.ed_ckpt
    if os.path.exists(ed_path):
        print(f"[INFO] Loading pre-trained Emotion Discriminator from {ed_path}")
        ckpt = torch.load(ed_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        D_emotion.load_state_dict(state_dict, strict=False)
    else:
        print(f"[WARN] ED checkpoint not found at {ed_path}. ED will be random!")

    # Freeze Emotion Discriminator
    for param in D_emotion.parameters():
        param.requires_grad = False
    D_emotion.eval()

    # --- 4. Optimizers ---
    opt_G = optim.Adam(
        list(G.parameters()) + list(E_num.parameters()), 
        lr=float(cfg['LR_G']), 
        betas=(cfg.get('BETA1', 0.5), cfg.get('BETA2', 0.9))
    )
    opt_D = optim.Adam(
        D_discriminator.parameters(), 
        lr=float(cfg['LR_D']), 
        betas=(cfg.get('BETA1', 0.5), cfg.get('BETA2', 0.9))
    )

    # --- 5. Training Loop ---
    writer = SummaryWriter(log_dir=cfg['LOG_DIR'])
    os.makedirs(cfg['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(cfg['SAMPLE_DIR'], exist_ok=True)
    
    criterion_emo = nn.CrossEntropyLoss()
    lambda_gp = cfg.get('LAMBDA_GP', 10.0)
    lambda_emotion = cfg.get('LAMBDA_EMOTION', 1.0)
    critic_iters = cfg.get('CRITIC_ITERS', 5)

    print("Starting WGAN-GP Training with Emotion Guidance...")

    for epoch in range(1, cfg['EPOCHS'] + 1):
        G.train()
        E_num.train()
        D_discriminator.train()
        
        ep_loss_d = 0.0
        ep_loss_g = 0.0
        ep_loss_g_emo = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            notes_real, emot_batch, latent_batch, numeric_batch = batch
            bsize = notes_real.size(0)

            notes_real = notes_real.to(device)
            numeric_batch = numeric_batch.to(device)
            emot_idx = torch.tensor([emotion_to_index(e) for e in emot_batch], dtype=torch.long, device=device)
            
            encoder_latent = None
            if latent_batch is not None:
                encoder_latent = (torch.from_numpy(latent_batch) if isinstance(latent_batch, np.ndarray) else torch.tensor(np.stack(latent_batch))).float().to(device)

            # ---------------------
            #  Train Discriminator (D)
            # ---------------------
            opt_D.zero_grad()

            with torch.no_grad():
                numeric_emb_d = E_num(numeric_batch)
                noise = torch.randn(bsize, cfg['NOISE_DIM'], device=device)
                gen_notes_d, _ = G(noise, encoder_latent, numeric_emb_d)
            
            # Real Score
            d_real = D_discriminator(notes_real, numeric_emb_d)
            
            # Fake Score
            d_fake = D_discriminator(gen_notes_d.detach(), numeric_emb_d)
            
            # Gradient Penalty
            gp = compute_gradient_penalty(D_discriminator, notes_real.data, gen_notes_d.data, numeric_emb_d, device)
            
            # Loss: This is the line that failed before.
            # d_fake and d_real are now tensors, so torch.mean will work.
            loss_d = torch.mean(d_fake) - torch.mean(d_real) + (lambda_gp * gp)
            
            loss_d.backward()
            opt_D.step()
            ep_loss_d += loss_d.item()

            
            # ---------------------
            #  Train Generator (G)
            # ---------------------
            if (batch_idx + 1) % critic_iters == 0:
                opt_G.zero_grad()
                
                # Get fresh embeddings with gradients enabled
                numeric_emb_g = E_num(numeric_batch)
                
                # We need fresh noise
                noise_g = torch.randn(bsize, cfg['NOISE_DIM'], device=device)
                
                # --- THIS IS THE FIX ---
                # Capture BOTH notes and the internal latent vector from the Generator
                gen_notes_g, gen_latent_g = G(noise_g, encoder_latent, numeric_emb_g)
                
                # 1. Adversarial Loss (Fool the Discriminator)
                d_fake_g = D_discriminator(gen_notes_g, numeric_emb_g)
                loss_g_adv = -torch.mean(d_fake_g)
                
                # 2. Emotion Loss (Fool the ED)
                # Check the ED's required input mode from its config
                ed_input_mode = ed_cfg.get('input_mode', 'notes')
                
                if ed_input_mode == 'latent':
                    # Pass the Generator's internal latent vector
                    ed_input = gen_latent_g
                else: # 'notes'
                    # Pass the generated notes
                    ed_input = gen_notes_g

                # Get logits from the (frozen) Emotion Discriminator
                ed_logits = D_emotion(ed_input)
                
                loss_g_emo_cls = criterion_emo(ed_logits, emot_idx)
                
                # Total G Loss
                loss_g = loss_g_adv + (lambda_emotion * loss_g_emo_cls)
                
                loss_g.backward()
                opt_G.step()
                
                ep_loss_g += loss_g_adv.item()
                ep_loss_g_emo += loss_g_emo_cls.item()
        # --- Epoch Logging ---
        
        steps = len(train_loader)
        g_steps = max(1, steps // critic_iters)
        
        print(f"Epoch {epoch}/{cfg['EPOCHS']} | "
              f"D_loss: {ep_loss_d/steps:.4f} | "
              f"G_adv: {ep_loss_g/g_steps:.4f} | "
              f"G_emo: {ep_loss_g_emo/g_steps:.4f}")
              
        writer.add_scalar("Loss/Critic", ep_loss_d/steps, epoch)
        writer.add_scalar("Loss/Generator_Adv", ep_loss_g/g_steps, epoch)
        writer.add_scalar("Loss/Generator_Emo", ep_loss_g_emo/g_steps, epoch)

        # --- Checkpointing & Sampling (Omitted for brevity, same as before) ---
        if epoch % cfg.get('SAVE_FREQ', 5) == 0:
            save_path = os.path.join(cfg['CHECKPOINT_DIR'], f"gan_epoch{epoch:04d}.pth")
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D_discriminator.state_dict(),
                'E_num': E_num.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict()
            }, save_path)

    # Final Save
    torch.save({
        'G': G.state_dict(),
        'E_num': E_num.state_dict()
    }, os.path.join(cfg['CHECKPOINT_DIR'], "gan_final.pth"))
    
    writer.close()
    print("Training Complete.")