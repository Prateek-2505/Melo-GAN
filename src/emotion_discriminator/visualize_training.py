"""
Training script for the Emotion Discriminator (Phase 1A).

Uses:
  - ed_model.py        (model architecture)
  - ed_dataset.py      (dataset + dataloader builder)
  - ed_config.yaml     (hyperparameters and paths)

Outputs:
  - Best checkpoint saved as <checkpoint_dir>/<save_name>
  - Logs for each epoch (loss, accuracy)

Supports:
  - latent or notes input mode
  - early stopping
  - ReduceLROnPlateau (optional)
  - Automatic device placement
"""

import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .ed_model import EmotionDiscriminator
from .ed_dataset import build_dataloader


def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()


def save_checkpoint(model, optimizer, epoch, cfg, is_best=False):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(
        cfg["checkpoint_dir"],
        cfg["save_name"] if is_best else f"ed_epoch{epoch}.pth"
    )
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg
    }
    torch.save(ckpt, save_path)
    return save_path


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_acc = 0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        # accumulate metrics
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    opt_name = cfg["optimizer"]["name"].lower()
    lr = float(cfg["optimizer"]["lr"])
    wd = cfg["optimizer"].get("weight_decay", 0)
    betas = tuple(cfg["optimizer"].get("betas", [0.9, 0.999]))

    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    elif opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer {opt_name}")


def build_scheduler(optimizer, cfg):
    if "scheduler" not in cfg:
        return None

    sch = cfg["scheduler"]
    name = sch.get("name", None)
    if name is None:
        return None

    if name.lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch.get("mode", "min"),
            factor=sch.get("factor", 0.5),
            patience=sch.get("patience", 5),
            threshold=sch.get("threshold", 1e-4),
            # verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler {name}")


def main(cfg_path="config/ed_config.yaml"):
    cfg = load_yaml(cfg_path)

    # device
    device = torch.device(cfg.get("device", "cpu"))

    # dataloaders
    train_loader = build_dataloader(cfg, split="train", shuffle=True)
    val_loader = build_dataloader(cfg, split="val", shuffle=False)

    # model
    model = EmotionDiscriminator(cfg).to(device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # training params
    epochs = cfg.get("num_epochs", 50)
    early_stop_pat = cfg.get("early_stopping_patience", 10)
    best_metric = float("inf") if cfg["metric_for_best"] == "val_loss" else 0
    best_epoch = 0

    print("\n🚀 Starting Emotion Discriminator Training")
    print("Input mode:", cfg["input_mode"])
    print("Device:", device)
    print("Epochs:", epochs)
    print("Best metric target:", cfg["metric_for_best"])
    print("-" * 80)

    # training loop
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        # scheduler step
        if scheduler:
            if cfg["metric_for_best"] == "val_loss":
                scheduler.step(val_loss)
            else:
                scheduler.step(val_acc)

        # decide if best
        if cfg["metric_for_best"] == "val_loss":
            current_metric = val_loss
            better = current_metric < best_metric
        else:
            current_metric = val_acc
            better = current_metric > best_metric

        if better:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, cfg, is_best=True)
            status = "🔥 New Best"
        else:
            status = ""

        # epoch log
        print(
            f"[Epoch {epoch:03d}] "
            f"Train-Loss={train_loss:.4f}, Train-Acc={train_acc:.3f} | "
            f"Val-Loss={val_loss:.4f}, Val-Acc={val_acc:.3f} {status}"
        )

        # early stopping
        if epoch - best_epoch >= early_stop_pat:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

        # save periodic checkpoints
        if epoch % cfg.get("save_freq", 5) == 0:
            save_checkpoint(model, optimizer, epoch, cfg, is_best=False)

    print("\n🎯 Training Completed.")
    print("Best epoch:", best_epoch)
    print("Best metric:", best_metric)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ed_config.yaml")
    args = parser.parse_args()

    main(args.config)