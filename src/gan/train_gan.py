#!/usr/bin/env python3
"""
Train GAN using AE encoder either as conditioning vector or warm-start decoder.

This version expects your config to include:
 - SPLITS_DIR (e.g. "data/splits")
 - TRAIN_SPLIT (CSV path relative to repo or absolute)
 - VAL_SPLIT
 - ENCODER_FEATS_TRAIN / ENCODER_FEATS_VAL (paths to .npy produced by encoder)
 - PROCESSED_DIR (e.g. "data/processed")
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

from src.gan.models import Generator, Discriminator
from src.gan.dataset import GANDataset
from src.gan.utils import seed_everything, weights_init, load_ae_decoder_into_generator, emotion_to_index

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def prepare_dataset(cfg, split_csv, latent_feats_path=None):
    # allow splits dir to be specified in config
    splits_dir = cfg.get('SPLITS_DIR', 'data/splits')
    # If user passed a relative filename, try to resolve under splits_dir
    if not os.path.isabs(split_csv) and not os.path.exists(split_csv):
        candidate = os.path.join(splits_dir, split_csv)
        if os.path.exists(candidate):
            split_csv = candidate
    split_name = Path(split_csv).stem
    notes_npy = os.path.join(splits_dir, split_name, "notes.npy")
    emo_npy = os.path.join(splits_dir, split_name, "emotion.npy")
    latent_feats = None
    if latent_feats_path and os.path.exists(latent_feats_path):
        latent_feats = np.load(latent_feats_path)
    # processed_dir from config
    processed_dir = cfg.get('PROCESSED_DIR', 'data/processed')
    ds = GANDataset(split_csv, processed_dir=processed_dir,
                    notes_npy=notes_npy if os.path.exists(notes_npy) else None,
                    emotion_npy=emo_npy if os.path.exists(emo_npy) else None,
                    latent_feats=latent_feats)
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gan_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.get("SEED", 42))
    device = torch.device(cfg.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

    # prepare dataset
    train_ds = prepare_dataset(cfg, cfg['TRAIN_SPLIT'], latent_feats_path=cfg.get('ENCODER_FEATS_TRAIN', None))
    val_ds = prepare_dataset(cfg, cfg['VAL_SPLIT'], latent_feats_path=cfg.get('ENCODER_FEATS_VAL', None))

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.get('BATCH_SIZE', 32), shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get('BATCH_SIZE', 32), shuffle=False, num_workers=2)

    print("Train set size:", len(train_ds), "Val set size:", len(val_ds))

    # instantiate models
    mode = cfg.get('INTEGRATION_MODE', 'conditioning')
    G = Generator(noise_dim=cfg['NOISE_DIM'], latent_dim=cfg['LATENT_DIM'], mode=mode, hidden=cfg.get('GEN_HIDDEN',512), max_notes=cfg['MAX_NOTES'], note_dim=cfg['NOTE_DIM']).to(device)
    D = Discriminator(max_notes=cfg['MAX_NOTES'], note_dim=cfg['NOTE_DIM']).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    # If warm_start, try loading AE decoder weights
    if mode == "warm_start":
        ok = load_ae_decoder_into_generator(cfg.get('AE_FULL_CKPT', ''), G)
        if not ok:
            print("[WARN] warm_start failed, switching to conditioning mode (requires encoder_feats).")
            mode = "conditioning"
            G.mode = "conditioning"

    # optimizers
    opt_G = optim.Adam(G.parameters(), lr=float(cfg['LR_G']), betas=(cfg.get('BETA1',0.5), 0.999), weight_decay=float(cfg.get('WEIGHT_DECAY', 0.0)))
    opt_D = optim.Adam(D.parameters(), lr=float(cfg['LR_D']), betas=(cfg.get('BETA1',0.5), 0.999), weight_decay=float(cfg.get('WEIGHT_DECAY', 0.0)))

    adversarial_loss = nn.BCEWithLogitsLoss()
    emotion_loss = nn.CrossEntropyLoss()

    # training loop
    os.makedirs(cfg['CHECKPOINT_DIR'], exist_ok=True)
    os.makedirs(cfg['LOG_DIR'], exist_ok=True)
    os.makedirs(cfg['SAMPLE_DIR'], exist_ok=True)

    for epoch in range(1, cfg['EPOCHS'] + 1):
        G.train(); D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            notes_batch, emot_batch, latent_batch = batch  # notes:(B,MAX_NOTES,4), emot: label or mood, latent: (B,LATENT_DIM) or None

            # convert emot to indices (emotion_to_index handles many types)
            emot_idx = torch.tensor([emotion_to_index(e) for e in emot_batch], dtype=torch.long, device=device)

            notes_batch = notes_batch.to(device)

            # prepare latent_feats (for conditioning) if provided by dataset
            if latent_batch is not None:
                # latent_batch may be a numpy array (if dataset used npy) or list; handle robustly
                if isinstance(latent_batch, np.ndarray):
                    encoder_latent = torch.from_numpy(latent_batch).float().to(device)
                else:
                    encoder_latent = torch.tensor(np.stack(latent_batch), dtype=torch.float32, device=device)
            else:
                encoder_latent = None

            bsize = notes_batch.size(0)
            # --------------------------
            # Train Discriminator
            # --------------------------
            opt_D.zero_grad()
            # Real
            rf_real_logits, emo_logits_real, _ = D(notes_batch)
            real_targets = torch.ones_like(rf_real_logits, device=device)
            loss_real = adversarial_loss(rf_real_logits, real_targets)
            # emotion classification loss on real
            loss_emo_real = emotion_loss(emo_logits_real, emot_idx)
            # Fake
            noise = torch.randn(bsize, cfg['NOISE_DIM'], device=device)
            if mode == "conditioning":
                if encoder_latent is None:
                    raise RuntimeError("conditioning mode requires encoder latent vectors in dataset (provide ENCODER_FEATS_TRAIN in config).")
                gen_notes, _ = G(noise, encoder_latent)
            else:
                gen_notes, _ = G(noise, None)
            rf_fake_logits, emo_logits_fake, _ = D(gen_notes.detach())
            fake_targets = torch.zeros_like(rf_fake_logits, device=device)
            loss_fake = adversarial_loss(rf_fake_logits, fake_targets)
            # discriminator total
            loss_D = loss_real + loss_fake + loss_emo_real * cfg.get('LAMBDA_EMOTION', 1.0)
            loss_D.backward()
            opt_D.step()

            # --------------------------
            # Train Generator
            # --------------------------
            opt_G.zero_grad()
            noise = torch.randn(bsize, cfg['NOISE_DIM'], device=device)
            if mode == "conditioning":
                gen_notes, gen_latent = G(noise, encoder_latent)
            else:
                gen_notes, gen_latent = G(noise, None)
            rf_logits_fake, emo_logits_on_fake, _ = D(gen_notes)
            # adversarial loss: want D to say real
            loss_G_adv = adversarial_loss(rf_logits_fake, torch.ones_like(rf_logits_fake, device=device))
            # emotion loss: encourage ED to predict the target emotion for generated sample
            loss_G_emo = emotion_loss(emo_logits_on_fake, emot_idx)
            # optional recon loss: if you want generator output to be near AE recon, not used by default
            loss_G = loss_G_adv + cfg.get('LAMBDA_EMOTION', 1.0) * loss_G_emo
            loss_G.backward()
            opt_G.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()

        epoch_g_loss /= (batch_idx + 1)
        epoch_d_loss /= (batch_idx + 1)
        print(f"Epoch {epoch}/{cfg['EPOCHS']}  G_loss={epoch_g_loss:.6f}  D_loss={epoch_d_loss:.6f}")

        # save sample outputs every SAVE_FREQ epochs
        if epoch % cfg.get('SAVE_FREQ', 5) == 0:
            G.eval()
            with torch.no_grad():
                n_per_em = 2
                for emo_idx in range(4):
                    for i in range(n_per_em):
                        z = torch.randn(1, cfg['NOISE_DIM'], device=device)
                        if mode == "conditioning":
                            if getattr(train_ds, 'latent_feats', None) is not None:
                                # choose random latent (fallback random normal if not available)
                                try:
                                    rand_idx = np.random.randint(0, train_ds.latent_feats.shape[0])
                                    latent_vec = torch.from_numpy(train_ds.latent_feats[rand_idx]).unsqueeze(0).to(device)
                                except Exception:
                                    latent_vec = torch.randn(1, cfg['LATENT_DIM'], device=device)
                            else:
                                latent_vec = torch.randn(1, cfg['LATENT_DIM'], device=device)
                            gen_out, _ = G(z, latent_vec)
                        else:
                            gen_out, _ = G(z, None)
                        out_np = gen_out.cpu().numpy()[0]
                        sample_path = os.path.join(cfg['SAMPLE_DIR'], f"epoch{epoch}_emo{emo_idx}_s{i}.npy")
                        np.save(sample_path, out_np)
            G.train()

        # save checkpoints
        if epoch % cfg.get('SAVE_FREQ', 5) == 0:
            torch.save({'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict()}, os.path.join(cfg['CHECKPOINT_DIR'], f"gan_epoch{epoch}.pth"))

    # final save
    torch.save({'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict()}, os.path.join(cfg['CHECKPOINT_DIR'], f"gan_final.pth"))
    print("Training complete.")
