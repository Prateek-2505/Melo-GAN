"""
Comprehensive analysis and diagnostics for Emotion Discriminator training.

Save this file as:
  src/emotion_discriminator/analysis.py

Usage examples:
  # Basic evaluation of best checkpoint + plots
  python -m src.emotion_discriminator.analysis --config config/ed_config.yaml --checkpoint experiments/ed/models/ed_best.pth

  # Run full diagnostics including one minibatch gradient/weight norms
  python -m src.emotion_discriminator.analysis --config config/ed_config.yaml --checkpoint experiments/ed/models/ed_best.pth --check-grads

Features implemented:
- Load config and history (experiments/ed/logs/ed_training_history.json by default)
- Load specified checkpoint (supports map_location)
- Evaluate on val set: confusion matrix, per-class precision/recall/F1, balanced accuracy
- Print class counts for train and val splits
- Plot training/validation loss & accuracy curves (saves PDF/PNG)
- Compute weight norms and (optionally) gradient norms on one batch
- Inspect optimizer & scheduler state in checkpoint (if present)
- Optional: run N repeated evals with different seeds to estimate metric variance

Notes:
- This script depends on your package layout. It imports `build_dataloader` and `EmotionDiscriminator` using absolute imports consistent with
  running from project root with `python -m src.emotion_discriminator.analysis`.

"""

import os
import argparse
import json
import math
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Project-specific imports (adjust if your package structure differs)
try:
    from .ed_dataset import build_dataloader
    from .ed_model import EmotionDiscriminator
except Exception:
    # fallback for direct execution (useful during development)
    from src.emotion_discriminator.ed_dataset import build_dataloader
    from src.emotion_discriminator.ed_model import EmotionDiscriminator


def load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_history(cfg):
    default = os.path.join('experiments', 'ed', 'logs', 'ed_training_history.json')
    path = cfg.get('history_path', default)
    if not os.path.exists(path):
        print(f"[WARN] history file not found at {path}")
        return None
    return json.load(open(path, 'r'))


def plot_history(history, out_dir='experiments/ed/analysis'):
    os.makedirs(out_dir, exist_ok=True)
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        plt.figure()
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        out = os.path.join(out_dir, 'loss_curves.png')
        plt.savefig(out)
        print('[INFO] Saved', out)
        plt.close()
    # Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        plt.figure()
        plt.plot(history['train_acc'], label='train_acc')
        plt.plot(history['val_acc'], label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curves')
        out = os.path.join(out_dir, 'acc_curves.png')
        plt.savefig(out)
        print('[INFO] Saved', out)
        plt.close()


def print_class_counts(cfg):
    for split in ['train', 'val']:
        loader = build_dataloader(cfg, split=split, shuffle=False)
        # We only need labels, so iterate quickly
        counts = Counter()
        for b in loader:
            ys = b['y'].cpu().numpy()
            counts.update(map(int, ys.tolist()))
        total = sum(counts.values())
        print(f"\n[INFO] {split} split: {total} samples")
        for k in sorted(counts.keys()):
            print(f"  class {k}: {counts[k]} ({counts[k]/total:.3f})")


def evaluate_model(model, loader, device, return_preds=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            x = b['x'].to(device)
            y = b['y'].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    bal = balanced_accuracy_score(all_labels, all_preds)
    # print human readable
    print('\n[RESULTS] Classification Report:')
    for k, v in report.items():
        if k.isdigit() or k in ['macro avg', 'weighted avg']:
            print(f"{k}: precision={v['precision']:.4f}, recall={v['recall']:.4f}, f1={v['f1-score']:.4f}, support={int(v['support'])}")
    print('\nConfusion Matrix:\n', cm)
    print(f'Balanced Accuracy: {bal:.4f}')

    if return_preds:
        return all_labels, all_preds
    return report, cm, bal


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model_from_ckpt(cfg, ckpt_path, device):
    model = EmotionDiscriminator(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        # backward compatibility: maybe the checkpoint stored state_dict directly
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            raise RuntimeError('Could not load model state from checkpoint')
    return model, ckpt


def compute_weight_norms(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            norms[name] = p.data.norm().item()
    # Summarize
    total = sum(v for v in norms.values())
    print('\n[INFO] Weight norms (sum of param norms):', total)
    # print top few
    items = sorted(norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for n, v in items:
        print(f"  {n}: {v:.4f}")
    return norms


def compute_grad_norms_on_one_batch(model, loader, device, criterion=nn.CrossEntropyLoss()):
    model.train()
    # grab one batch
    batch = next(iter(loader))
    x = batch['x'].to(device)
    y = batch['y'].to(device)
    # zero grads
    model.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.data.norm().item()
    # summarize
    total = sum(v for v in grad_norms.values())
    print('\n[INFO] Gradient norms (sum):', total)
    items = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for n, v in items:
        print(f"  {n}: {v:.6f}")
    return grad_norms


def inspect_optimizer_scheduler(ckpt):
    if 'optimizer' in ckpt:
        print('\n[INFO] Optimizer state found in checkpoint')
        try:
            opt_state = ckpt['optimizer']
            # print param group LR
            if 'param_groups' in opt_state:
                lrs = [g.get('lr', None) for g in opt_state['param_groups']]
                print('  param_group lrs:', lrs)
        except Exception as e:
            print('  failed to parse optimizer state:', e)
    else:
        print('\n[INFO] No optimizer state in checkpoint')

    if 'cfg' in ckpt:
        print('\n[INFO] cfg stored in checkpoint; printing important keys:')
        stored = ckpt['cfg']
        keys = ['optimizer', 'scheduler', 'num_epochs', 'save_freq']
        for k in keys:
            if k in stored:
                print(f"  {k}: {stored[k]}")

    if 'scheduler' in ckpt:
        print('\n[INFO] Scheduler state present')


def repeated_evals(cfg, ckpt_path, device, n=5):
    # Run repeated evaluations by shuffling val loader if shuffle=True or by setting different seeds
    res = []
    for i in range(n):
        seed = 1000 + i
        torch.manual_seed(seed); np.random.seed(seed)
        val_loader = build_dataloader(cfg, split='val', shuffle=False)
        model, _ = build_model_from_ckpt(cfg, ckpt_path, device)
        _, _, bal = evaluate_model(model, val_loader, device)
        res.append(bal)
    arr = np.array(res)
    print(f"\n[REPEATS] n={n}, mean_bal={arr.mean():.4f}, std={arr.std():.4f}")
    return arr


def main(args):
    cfg = load_yaml(args.config)
    history = load_history(cfg)
    if history is not None:
        plot_history(history, out_dir=args.out_dir)

    # Print class counts
    print_class_counts(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print('\n[INFO] Using device:', device)

    # Load model & checkpoint
    model, ckpt = build_model_from_ckpt(cfg, args.checkpoint, device)
    inspect_optimizer_scheduler(ckpt)

    # Evaluate on validation set
    val_loader = build_dataloader(cfg, split='val', shuffle=False)
    print('\n[INFO] Evaluating model on val set...')
    evaluate_model(model, val_loader, device)

    # Optionally run repeated evals
    if args.repeats > 1:
        repeated_evals(cfg, args.checkpoint, device, n=args.repeats)

    # Weight norms
    compute_weight_norms(model)

    # Optional gradient norms on one batch
    if args.check_grads:
        train_loader = build_dataloader(cfg, split='train', shuffle=True)
        compute_grad_norms_on_one_batch(model, train_loader, device)

    print('\n[INFO] Analysis complete. Plots and outputs are in', args.out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config/ed_config.yaml')
    p.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    p.add_argument('--out-dir', type=str, default=os.path.join('experiments', 'ed', 'analysis'))
    p.add_argument('--check-grads', action='store_true', help='Compute gradients on one training batch (requires train loader)')
    p.add_argument('--repeats', type=int, default=1, help='Run repeated evaluations to estimate variance')
    p.add_argument('--force-cpu', action='store_true', help='Force CPU even if CUDA available')
    args = p.parse_args()

    main(args)
