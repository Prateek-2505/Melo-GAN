# paths & dirs
import os
import glob
import pandas as pd
from .path_utils import ensure_dir

def resolve(cfg):
    processed_dir = cfg.get('PROCESSED_DIR', 'processed')
    splits_dir = cfg.get('SPLITS_DIR', 'splits')
    model_dir = cfg.get('CHECKPOINT_DIR', 'models/ae')
    log_dir = cfg.get('LOG_DIR', 'experiments/ae')
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(cfg.get('RECON_DIR','experiments/ae/reconstructions'))

    # load split file lists (expects CSVs with 'filename' column or plain filename lists)
    train_csv = os.path.join(splits_dir, 'train_split.csv')
    val_csv = os.path.join(splits_dir, 'val_split.csv')
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("train_split.csv or val_split.csv not found in splits/")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Build file lists (paths to .npz)
    def resolve_paths(df):
        files = []
        for _, row in df.iterrows():
            fname = row['npz_filename']
            # possible that processed files were saved as <stem>.npz
            candidates = [
                os.path.join(processed_dir, fname if fname.endswith('.npz') else os.path.splitext(fname)[0] + '.npz')
            ]
            # fallback: search processed dir for any containing filename stem
            if not os.path.exists(candidates[0]):
                stem = os.path.splitext(fname)[0]
                found = glob.glob(os.path.join(processed_dir, f"*{stem}*.npz"))
                if found:
                    candidates = [found[0]]
            if os.path.exists(candidates[0]):
                files.append(candidates[0])
            else:
                print("Warning: processed file for", fname, "not found. Skipping.")
        return files

    train_files = resolve_paths(train_df)
    val_files = resolve_paths(val_df)
    return train_files, val_files