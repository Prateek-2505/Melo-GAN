# src/gan/train_gan.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

# Project modules
from src.gan.models import Generator, Discriminator
from src.gan.feature_encoder import FeatureEncoder
from src.gan.utils import (
    seed_everything, weights_init, compute_gradient_penalty, 
    load_ae_decoder_into_generator, emotion_to_index
)
from src.emotion_discriminator.ed_model import EmotionDiscriminator
from src.emotion_discriminator.ed_dataset import build_dataloader

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(config_path, ed_config_path):
    # 1. Load Configs
    cfg = load_config(config_path)
    ed_cfg = load_config(ed_config_path)

    seed_everything(cfg.get('SEED', 42))
    device = torch.device(cfg.get('DEVICE', "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using main device: {device}")

    # 2. Prepare Data
    train_loader = build_dataloader(ed_cfg, split="train", shuffle=True)
    print(f"Train set size: {len(train_loader.dataset)}")

    # 3. Initialize Models
    numeric_dim = cfg.get('NUMERIC_INPUT_DIM', 6)
    embed_dim = cfg.get('ENCODER_OUT_DIM', 128)
    
    E_num = FeatureEncoder(
        in_dim=numeric_dim, 
        hidden_dims=cfg.get('ENCODER_HIDDEN', [64]), 
        out_dim=embed_dim
    ).to(device)

    G = Generator(
        noise_dim=cfg['NOISE_DIM'], 
        latent_dim=cfg['LATENT_DIM'], 
        mode=cfg.get('INTEGRATION_MODE', 'warm_start'), 
        hidden=cfg.get('GEN_HIDDEN', 512),
        max_notes=cfg['MAX_NOTES'],
        note_dim=cfg['NOTE_DIM'],
        numeric_embed_dim=embed_dim
    ).to(device)
    
    D = Discriminator(
        max_notes=cfg['MAX_NOTES'], 
        note_dim=cfg['NOTE_DIM'],
        emb_dim=cfg.get('DISC_EMB_DIM', 256),
        numeric_embed_dim=embed_dim
    ).to(device)

    # Apply weights init
    G.apply(weights_init)
    D.apply(weights_init)
    E_num.apply(weights_init)

    # 4. Load Emotion Discriminator (Frozen Judge)
    print(f"[INFO] Instantiating Emotion Discriminator using {os.path.basename(ed_config_path)}")
    ED_Judge = EmotionDiscriminator(ed_cfg).to(device)
    
    ed_ckpt_path = "data/models/ed/ed_best.pth"
    if os.path.exists(ed_ckpt_path):
        print(f"[INFO] Loading pre-trained Emotion Discriminator from {ed_ckpt_path}")
        ckpt = torch.load(ed_ckpt_path, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        ED_Judge.load_state_dict(state_dict)
    else:
        print(f"[WARN] No ED checkpoint found at {ed_ckpt_path}. Training will lack guidance!")
    
    ED_Judge.eval()
    for p in ED_Judge.parameters():
        p.requires_grad = False

    # 5. Optimizers
    lr = float(cfg.get('LR', 1e-4))
    b1 = float(cfg.get('BETA1', 0.5))
    b2 = float(cfg.get('BETA2', 0.9))
    
    opt_G = optim.Adam(list(G.parameters()) + list(E_num.parameters()), lr=lr, betas=(b1, b2))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))
    
    # --- LOGGING SETUP ---
    log_dir = cfg.get('LOG_DIR', 'experiments/gan/logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # JSON History File Path
    history_path = os.path.join(log_dir, "gan_training_history.json")
    
    # Initialize History Dict
    history = {
        "d_loss": [],
        "g_adv": [],
        "g_emo": []
    }
    # --- UPDATED RESUME LOGIC (SAFE MODE) ---
    start_epoch = 1
    ckpt_dir = cfg.get('CHECKPOINT_DIR', 'experiments/gan/checkpoints') #experiments/gan/checkpoints/gan_final.pth
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = os.path.join(ckpt_dir, "gan_final.pth")

    if os.path.exists(resume_path):
        print(f"[INFO] Checkpoint found at {resume_path}. Attempting to resume...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 1. Load Generator (Most Important)
            if 'G' in checkpoint:
                G.load_state_dict(checkpoint['G'])
                print("[INFO] Loaded Generator state.")
            else:
                print("[WARN] 'G' key missing in checkpoint. Generator starting fresh.")

            # 2. Load Encoder
            if 'E_num' in checkpoint:
                E_num.load_state_dict(checkpoint['E_num'])
                print("[INFO] Loaded Feature Encoder state.")

            # 3. Load Discriminator (Might be missing in old files)
            if 'D' in checkpoint:
                D.load_state_dict(checkpoint['D'])
                print("[INFO] Loaded Discriminator state.")
            else:
                print("[WARN] 'D' key missing (old checkpoint format). Discriminator starting fresh.")

            # 4. Load Optimizers
            if 'opt_G' in checkpoint: opt_G.load_state_dict(checkpoint['opt_G'])
            if 'opt_D' in checkpoint: opt_D.load_state_dict(checkpoint['opt_D'])
            
            # 5. Load Epoch
            if 'epoch' in checkpoint: 
                start_epoch = checkpoint['epoch'] + 1
                print(f"[INFO] Resuming from Epoch {start_epoch}")
            else:
                print("[INFO] Epoch info missing. Assuming Epoch 1 (but preserving weights).")

        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print("[INFO] Starting fresh to avoid corruption.")
    else:
        print("[INFO] No previous checkpoint found. Starting fresh training.")
    # --- END RESUME LOGIC ---

    # 6. Training Loop
    writer = SummaryWriter(log_dir=cfg.get('LOG_DIR', 'experiments/gan/logs'))
    epochs = cfg.get('EPOCHS', 100)
    lambda_gp = float(cfg.get('LAMBDA_GP', 10.0))
    lambda_emo = float(cfg.get('LAMBDA_EMO', 1.0))
    n_critic = int(cfg.get('N_CRITIC', 5))

    print("Starting WGAN-GP Training with Emotion Guidance...")
    
    target_vectors = {
        0: torch.tensor([1.0, 1.0, 0.8, 0.8, 0.5, 0.5]).to(device),       # Happy
        1: torch.tensor([-1.0, -1.0, -0.5, -0.5, -0.5, -0.5]).to(device), # Sad
        2: torch.tensor([1.0, -1.0, 1.0, 1.0, -0.8, 0.8]).to(device),     # Angry
        3: torch.tensor([-1.0, 1.0, -0.8, -0.8, 0.5, -0.5]).to(device)    # Calm
    }

    for epoch in range(start_epoch, epochs + 1):
        loop = tqdm(train_loader, leave=True)
        d_losses = []
        g_adv_losses = []
        g_emo_losses = []

        for i, batch in enumerate(loop):
            real_notes = batch['x'].to(device) # (B, notes, 4)
            labels = batch['y'].to(device)
            batch_size = real_notes.size(0)

            real_numeric_list = []
            for lbl in labels:
                l_idx = int(lbl.item())
                vec = target_vectors.get(l_idx, torch.zeros(6).to(device))
                real_numeric_list.append(vec)
            
            real_numeric_cond = torch.stack(real_numeric_list)
            jitter = torch.randn_like(real_numeric_cond) * 0.1
            real_numeric_emb = E_num(real_numeric_cond + jitter)

            # Train Discriminator
            opt_D.zero_grad()
            noise = torch.randn(batch_size, cfg['NOISE_DIM']).to(device)
            encoder_latent = torch.zeros(batch_size, cfg['LATENT_DIM']).to(device)
            
            fake_notes, _ = G(noise, encoder_latent, real_numeric_emb)

            real_validity = D(real_notes, real_numeric_emb)
            fake_validity = D(fake_notes.detach(), real_numeric_emb)

            d_loss_real = -torch.mean(real_validity)
            d_loss_fake = torch.mean(fake_validity)
            gp = compute_gradient_penalty(D, real_notes, fake_notes.detach(), real_numeric_emb, device)
            
            d_loss = d_loss_real + d_loss_fake + lambda_gp * gp
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            
            # -----------------
            #  Train Generator
            # -----------------
            
            if i % n_critic == 0:
                opt_G.zero_grad()
                
                # Generate Fakes
                rand_labels = torch.randint(0, 4, (batch_size,)).to(device)
                fake_cond_list = [target_vectors[l.item()] for l in rand_labels]
                fake_cond = torch.stack(fake_cond_list)
                fake_emb = E_num(fake_cond)
                
                gen_notes, _ = G(noise, encoder_latent, fake_emb)
                
                # 1. Adversarial Loss
                fake_validity = D(gen_notes, fake_emb)
                g_adv_loss = -torch.mean(fake_validity)
                
                # 2. Emotion Class Loss
                ed_logits = ED_Judge(gen_notes)
                g_emo_loss = nn.CrossEntropyLoss()(ed_logits, rand_labels)

                # 3. EXPLICIT FEATURE GUIDANCE (Pitch, Velocity, AND STEP)
                # Channel 0=Pitch, 1=Velocity, 3=Step (Timing)
                
                batch_pitch_means = torch.mean(gen_notes[:, :, 0], dim=1)
                batch_vel_means   = torch.mean(gen_notes[:, :, 1], dim=1)
                batch_step_means  = torch.mean(gen_notes[:, :, 3], dim=1) # NEW: Step mean

                # Create Targets
                pitch_targets = torch.zeros_like(batch_pitch_means)
                vel_targets   = torch.zeros_like(batch_vel_means)
                step_targets  = torch.zeros_like(batch_step_means)
                
                for idx, lbl in enumerate(rand_labels):
                    if lbl.item() in [0, 2]: # Happy/Angry (Fast & High)
                        pitch_targets[idx] = 0.15   # Pitch ~73
                        vel_targets[idx]   = 0.3    # Vel ~82
                        step_targets[idx]  = -0.6   # Step ~0.6 beats (Fast)
                    else: # Sad/Calm (Slow & Low)
                        pitch_targets[idx] = -0.15  # Pitch ~54
                        vel_targets[idx]   = -0.1   # Vel ~58
                        step_targets[idx]  = 0.2    # Step ~2.4 beats (Slow)
                
                # Loss Calculation
                g_pitch_loss = nn.MSELoss()(batch_pitch_means, pitch_targets)
                g_vel_loss   = nn.MSELoss()(batch_vel_means, vel_targets)
                g_step_loss  = nn.MSELoss()(batch_step_means, step_targets) # NEW

                # Combine Losses (Weight step heavily to fix it fast)
                g_total_loss = g_adv_loss + (lambda_emo * g_emo_loss) + \
                               (10.0 * g_pitch_loss) + (10.0 * g_vel_loss) + (20.0 * g_step_loss)
                
                g_total_loss.backward()
                opt_G.step()
                
                g_adv_losses.append(g_adv_loss.item())
                g_emo_losses.append(g_emo_loss.item())
            loop.set_description(f"Epoch {epoch}/{epochs}")
            loop.set_postfix(d_loss=d_loss.item(), g_adv=g_adv_loss.item() if i%n_critic==0 else 0)

        avg_d_loss = np.mean(d_losses)
        avg_g_adv = np.mean(g_adv_losses) if g_adv_losses else 0
        avg_g_emo = np.mean(g_emo_losses) if g_emo_losses else 0
        
        print(f"Epoch {epoch}/{epochs} | D_loss: {avg_d_loss:.4f} | G_adv: {avg_g_adv:.4f} | G_emo: {avg_g_emo:.4f}")
        
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Generator_Adv", avg_g_adv, epoch)
        writer.add_scalar("Loss/Generator_Emo", avg_g_emo, epoch)

        # Append to JSON history (used by plotting script)
        history['d_loss'].append(float(avg_d_loss))
        history['g_adv'].append(float(avg_g_adv))
        history['g_emo'].append(float(avg_g_emo))
        try:
            with open(history_path, 'w') as hf:
                json.dump(history, hf, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write history file: {e}")

        # Save Checkpoint with ALL keys for next time
        torch.save({
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'E_num': E_num.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict()
        }, resume_path)

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/gan_config.yaml")
    parser.add_argument("--ed_config", type=str, default="config/ed_config.yaml")
    args = parser.parse_args()
    
    main(args.config, args.ed_config)