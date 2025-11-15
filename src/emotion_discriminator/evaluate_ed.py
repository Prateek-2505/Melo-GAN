"""
evaluate_ed.py

Evaluation / inference utilities for the Emotion Discriminator (ED).

Features:
 - Load a checkpoint saved by train_ed.py
 - Evaluate on test (or val/train) split and print metrics
 - Save predictions CSV, classification report (text), and confusion matrix PNG
 - Optional: score generated samples in a directory (npy/npz) and save their emotion probabilities
 - Handles both 'latent' and 'notes' input modes based on config

Usage:
    python -m src.emotion_discriminator.evaluate_ed --config config/ed_config.yaml --ckpt data/models/ed/ed_best.pth

Author: ChatGPT (for MELO-GAN Phase 1A)
"""
import os
import yaml
import argparse
import time
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .ed_model import EmotionDiscriminator
from .ed_dataset import build_dataloader, EmotionDataset

# sklearn for metrics (ensure installed in your env)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device = torch.device("cpu")) -> dict:
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model_from_cfg(cfg: dict, ckpt: Optional[dict] = None, device: torch.device = torch.device("cpu")) -> EmotionDiscriminator:
    model = EmotionDiscriminator(cfg)
    model.to(device)
    if ckpt is not None:
        state = ckpt.get("model", ckpt)  # accept direct state_dict or full ckpt
        model.load_state_dict(state)
    model.eval()
    return model


def run_inference_on_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    ys_true = []
    ys_pred = []
    probs_list = []
    metas = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            meta = batch.get("meta", None)

            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            ys_true.append(y)
            ys_pred.append(preds)
            probs_list.extend(probs.tolist())
            metas.extend(meta if meta is not None else [{}] * len(y))

    y_true = np.concatenate(ys_true, axis=0)
    y_pred = np.concatenate(ys_pred, axis=0)
    return y_true, y_pred, probs_list, metas


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_png: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_txt: str):
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    with open(out_txt, "w") as f:
        f.write(report)
    return report


def save_predictions_csv(y_true: np.ndarray, y_pred: np.ndarray, probs_list: List[List[float]], metas: List[dict], out_csv: str, labels: List[str]):
    rows = []
    for i in range(len(y_true)):
        row = {
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
            "true_label_name": labels[int(y_true[i])] if 0 <= int(y_true[i]) < len(labels) else str(y_true[i]),
            "pred_label_name": labels[int(y_pred[i])] if 0 <= int(y_pred[i]) < len(labels) else str(y_pred[i]),
        }
        # add probs as separate columns
        for j, p in enumerate(probs_list[i]):
            row[f"prob_{j}_{labels[j]}"] = float(p)
        # include meta keys (flatten small subset)
        meta = metas[i] if metas and i < len(metas) else {}
        if isinstance(meta, dict):
            for k, v in meta.items():
                # avoid huge blobs
                try:
                    row[f"meta_{k}"] = str(v)
                except Exception:
                    row[f"meta_{k}"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def score_generated_samples_dir(model: torch.nn.Module, gen_dir: str, cfg: dict, device: torch.device) -> pd.DataFrame:
    """
    Score generated samples found in gen_dir.
    Accepts .npy (latent vectors) or .npz (notes arrays).
    Returns a DataFrame with filename, top_pred, top_prob, probs...
    """
    files = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith((".npy", ".npz"))])
    out_rows = []
    labels = cfg.get("labels", ["happy", "sad", "angry", "calm"])
    for p in files:
        try:
            if p.endswith(".npy"):
                arr = np.load(p)
                x = torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0).to(device)
            else:
                # .npz expected to contain 'notes' or arr_0
                data = np.load(p, allow_pickle=True)
                if "notes" in data:
                    notes = data["notes"]
                else:
                    notes = data.get("arr_0", None)
                    if notes is None:
                        notes = data[list(data.files)[0]]
                # pad/truncate to max_notes if necessary
                maxn = int(cfg.get("max_notes", 512))
                note_dim = int(cfg.get("note_dim", 4))
                notes = np.asarray(notes, dtype=np.float32)
                if notes.ndim == 1:
                    notes = notes.reshape(-1, note_dim)
                T = notes.shape[0]
                if T >= maxn:
                    notes = notes[:maxn]
                else:
                    pad = np.zeros((maxn - T, note_dim), dtype=np.float32)
                    notes = np.concatenate([notes, pad], axis=0)
                x = torch.from_numpy(notes).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = int(probs.argmax())
                out_rows.append({
                    "file": os.path.basename(p),
                    "pred_label": pred,
                    "pred_label_name": labels[pred] if pred < len(labels) else str(pred),
                    "top_prob": float(probs[pred]),
                    **{f"prob_{i}_{labels[i]}": float(probs[i]) for i in range(len(probs))}
                })
        except Exception as e:
            out_rows.append({"file": os.path.basename(p), "error": str(e)})
    return pd.DataFrame(out_rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ed_config.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (ed_best.pth or ed_epochXX.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--gen_dir", type=str, default=None, help="Optional: directory of generated samples to score")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional override for output dir (defaults to cfg.log_dir)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    labels = cfg.get("labels", ["happy", "sad", "angry", "calm"])
    out_base = args.out_dir or cfg.get("log_dir", "data/experiments/ed")
    os.makedirs(out_base, exist_ok=True)

    device = torch.device(args.device or cfg.get("device", "cpu"))

    print(f"Loading checkpoint {args.ckpt} on device {device} ...")
    ckpt = load_checkpoint(args.ckpt, device=device)
    model = build_model_from_cfg(cfg, ckpt=ckpt, device=device)

    print("Building dataloader for split:", args.split)
    loader = build_dataloader(cfg, split=args.split, shuffle=False)

    print("Running inference...")
    t0 = time.time()
    y_true, y_pred, probs_list, metas = run_inference_on_loader(model, loader, device)
    t1 = time.time()
    print(f"Inference done in {t1 - t0:.2f}s. Samples: {len(y_true)}")

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    # save confusion matrix
    cm_png = os.path.join(out_base, f"confusion_{args.split}.png")
    plot_and_save_confusion_matrix(y_true, y_pred, labels, cm_png)
    print("Saved confusion matrix to", cm_png)

    # save classification report
    report_txt = os.path.join(out_base, f"classification_report_{args.split}.txt")
    report = save_classification_report(y_true, y_pred, labels, report_txt)
    print("Saved classification report to", report_txt)
    print(report)

    # save predictions csv
    preds_csv = os.path.join(out_base, f"predictions_{args.split}.csv")
    df_preds = save_predictions_csv(y_true, y_pred, probs_list, metas, preds_csv, labels)
    print("Saved predictions to", preds_csv)

    # optionally score generated samples
    if args.gen_dir:
        print("Scoring generated samples in", args.gen_dir)
        df_gen = score_generated_samples_dir(model, args.gen_dir, cfg, device)
        gen_out = os.path.join(out_base, "generated_samples_scored.csv")
        df_gen.to_csv(gen_out, index=False)
        print("Saved generated samples scores to", gen_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
