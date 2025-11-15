"""
visualize_training.py

Parse the training log produced by train_ed.py (via scripts/run_ed.sh)
and plot training curves: train/val loss and train/val accuracy.

Usage:
  python -m src.emotion_discriminator.visualize_training --logfile path/to/train_ed_YYYYMMDD_HHMMSS.log --out_dir data/experiments/ed

If the script cannot parse the log, it will attempt to locate any 'training_log.csv' in the same directory.
"""
import re
import argparse
import os
import sys
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import csv

EPOCH_LINE_RE = re.compile(
    r"\[Epoch\s+(\d+)\] .*Train-Loss=([0-9\.eE+-]+), Train-Acc=([0-9\.]+) \| Val-Loss=([0-9\.eE+-]+), Val-Acc=([0-9\.]+)"
)


def parse_log_file(logfile: str) -> Dict[str, List[float]]:
    """
    Parse logfile and extract per-epoch metrics.
    Returns a dict with keys: epoch, train_loss, train_acc, val_loss, val_acc
    """
    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    with open(logfile, "r", errors="replace") as f:
        for line in f:
            m = EPOCH_LINE_RE.search(line)
            if m:
                epoch = int(m.group(1))
                tloss = float(m.group(2))
                tacc = float(m.group(3))
                vloss = float(m.group(4))
                vacc = float(m.group(5))
                metrics["epoch"].append(epoch)
                metrics["train_loss"].append(tloss)
                metrics["train_acc"].append(tacc)
                metrics["val_loss"].append(vloss)
                metrics["val_acc"].append(vacc)
    return metrics


def save_csv(metrics: Dict[str, List[float]], out_csv: str):
    keys = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    rows = zip(*(metrics[k] for k in keys))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow(r)


def plot_metrics(metrics: Dict[str, List[float]], out_png: str):
    epochs = metrics["epoch"]
    if len(epochs) == 0:
        raise RuntimeError("No epoch lines parsed from log. Ensure train_ed.py printed lines in the expected format.")
    # Loss
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ED Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    base, ext = os.path.splitext(out_png)
    loss_png = base + "_loss.png"
    plt.savefig(loss_png, dpi=150)
    plt.close()

    # Accuracy
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, metrics["train_acc"], label="Train Acc")
    plt.plot(epochs, metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ED Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    acc_png = base + "_acc.png"
    plt.savefig(acc_png, dpi=150)
    plt.close()

    # Combined figure
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics["train_acc"], label="Train Acc")
    plt.plot(epochs, metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return loss_png, acc_png, out_png


def find_training_csv(logfile_dir: str):
    # fallback: look for training_log.csv
    candidate = os.path.join(logfile_dir, "training_log.csv")
    if os.path.exists(candidate):
        # load CSV expected to have columns epoch,train_loss,train_acc,val_loss,val_acc
        metrics = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        import csv
        with open(candidate, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                metrics["epoch"].append(int(row.get("epoch", 0)))
                metrics["train_loss"].append(float(row.get("train_loss", 0.0)))
                metrics["train_acc"].append(float(row.get("train_acc", 0.0)))
                metrics["val_loss"].append(float(row.get("val_loss", 0.0)))
                metrics["val_acc"].append(float(row.get("val_acc", 0.0)))
        return metrics
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, required=True, help="Path to train log file created by run_ed.sh")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save result PNG(s). Defaults to logfile directory")
    parser.add_argument("--out_name", type=str, default="training_curves.png", help="Base name for saved combined PNG")
    args = parser.parse_args()

    logfile = args.logfile
    if not os.path.exists(logfile):
        print("Logfile not found:", logfile)
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(logfile) or "."
    os.makedirs(out_dir, exist_ok=True)

    metrics = parse_log_file(logfile)
    if not metrics["epoch"]:
        # fallback to CSV in same dir
        fallback = find_training_csv(os.path.dirname(logfile))
        if fallback:
            metrics = fallback
        else:
            print("No epoch lines parsed from logfile and no training_log.csv found in same dir.")
            sys.exit(1)

    csv_out = os.path.join(out_dir, "training_metrics_parsed.csv")
    save_csv(metrics, csv_out)
    combined_png = os.path.join(out_dir, args.out_name)
    loss_png, acc_png, combined_png = plot_metrics(metrics, combined_png)

    print("Saved parsed metrics to:", csv_out)
    print("Saved loss plot to:", loss_png)
    print("Saved accuracy plot to:", acc_png)
    print("Saved combined plot to:", combined_png)


if __name__ == "__main__":
    main()
