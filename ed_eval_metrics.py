import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
import os               # <-- added
import json             # <-- added

# --- Import your team's code ---
from src.emotion_discriminator.ed_model import EmotionDiscriminator 

# --- CONFIGURATION & PATHS ---
# These paths assume you are running from the 'Melo-GAN' folder.
ROOT_PATH = Path("data")
SPLITS_PATH = ROOT_PATH / "splits"
PROCESSED_PATH = ROOT_PATH / "processed"
ED_CONFIG_PATH = Path("config") / "ed_config.yaml"
ED_CKPT_PATH = ROOT_PATH / "models" / "ed" / "ed_best.pth"

# Load config to get MAX_NOTES, etc.
with open(ED_CONFIG_PATH) as f:
    ed_cfg = yaml.safe_load(f)

# --- Helper Dataset for Testing ---
class EDTestDataset(Dataset):
    """Loads real preprocessed data for standalone evaluation."""
    def __init__(self, split_csv_path, processed_dir=PROCESSED_PATH):
        self.df = pd.read_csv(split_csv_path)
        self.processed_dir = processed_dir
        self.EMOTION_MAP = {'happy': 0, 'sad': 1, 'angry': 2, 'calm': 3}
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # This path must be generated correctly from the manifest path
        # Assuming your preprocessing script saved it as: source_filename.npz
        original_stem = Path(row['full_path']).stem
        npz_filename = f"{row['source']}_{original_stem}.npz"
        npz_path = self.processed_dir / npz_filename
        
        try:
            with np.load(npz_path) as data:
                # Load normalized notes array
                notes = data['notes'].astype(np.float32)
            
            # Load true label
            label = self.EMOTION_TO_LABEL.get(row['emotion'], 3)
            
            return torch.tensor(notes, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        except Exception:
            return None, None # Skip corrupt/missing files

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    return torch.utils.data.default_collate(batch)


# ================================================================
# 1. STANDALONE EVALUATION (ECA & AUC)
# ================================================================

def evaluate_ed_standalone(ed_model, device, test_csv_path):
    """Calculates ECA and AUC on a real, unseen dataset (Test Split)."""
    
    test_dataset = EDTestDataset(test_csv_path)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=collate_fn_skip_none, 
        num_workers=2
    )

    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    ed_model.eval()
    with torch.no_grad():
        for notes, targets in tqdm(test_loader, desc="Evaluating ED on Test Set"):
            if notes is None: continue

            notes = notes.to(device)
            targets = targets.to(device)
            
            # Get model outputs (Logits)
            logits = ed_model(notes)
            probabilities = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate Metrics
    acc = accuracy_score(all_targets, all_predictions)
    
    try:
        # AUC must be calculated using the probabilities
        # multi_class='ovr' (one-vs-rest) is standard for multi-class AUC
        auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
    except ValueError:
        # Happens if only one class is present in the batch, which shouldn't happen here
        auc = np.nan 

    return acc, auc, confusion_matrix(all_targets, all_predictions)


# ================================================================
# 2. TRAINING METRIC (Generator Emotion Loss)
# ================================================================

def calculate_generator_emotion_loss(generated_notes_batch, intended_labels_batch, ed_model):
    """
    Calculates Loss_G_emo: The Cross-Entropy Loss used to guide the Generator.
    This simulates the training loop's loss calculation.
    """
    
    ed_model.eval() # Ensure ED is frozen
    
    # Cross-Entropy Loss is the standard for multi-class classification
    criterion_emo = nn.CrossEntropyLoss()
    
    # Get logits from the generated music
    ed_logits = ed_model(generated_notes_batch)
    
    # Calculate CrossEntropyLoss using the logits and the INTENDED labels
    # This loss tells the Generator: "How far away are you from the emotion I told you to make?"
    loss_g_emo = criterion_emo(ed_logits, intended_labels_batch)
    
    return loss_g_emo.item()


# ================================================================
# MAIN EXECUTION EXAMPLE
# ================================================================

if __name__ == "__main__":
    
    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ed_model = EmotionDiscriminator(ed_cfg).to(device)
    
    try:
        # Load weights
        ckpt = torch.load(ED_CKPT_PATH, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        ed_model.load_state_dict(state_dict, strict=False)
        print("Emotion Discriminator loaded for evaluation.")
    except Exception as e:
        print(f"FATAL: Could not load ED checkpoint at {ED_CKPT_PATH}. Skipping evaluation.")
        sys.exit()

    # --- 1. RUN STANDALONE EVALUATION (ECA & AUC) ---
    print("\n" + "="*50)
    print("RUNNING STANDALONE EVALUATION (ECA & AUC)")
    print("="*50)
    
    # Use the test split for final evaluation
    TEST_SPLIT_CSV = SPLITS_PATH / 'test_split.csv'
    
    acc, auc, conf_matrix = evaluate_ed_standalone(ed_model, device, TEST_SPLIT_CSV)
    
    print("\n" + "="*50)
    print(f"RESULTS: ED Performance on Unseen Test Data")
    print("="*50)
    print(f"Emotion Classification Accuracy (ECA): {acc:.2f} %")
    print(f"ROC AUC Score (Multi-Class):         {auc:.4f}")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(conf_matrix)
    print("="*50)

    # --- Save confusion matrix & metadata for plotting ---
    out_dir = Path("experiments/ed/logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to get class labels from config (fallback to common order)
    classes = ed_cfg.get("classes", ["happy", "sad", "angry", "calm"])

    # Save as numpy for reloading, and JSON for human readability
    np.save(out_dir / "ed_confusion_matrix.npy", conf_matrix)
    with open(out_dir / "ed_confusion_matrix.json", "w") as jf:
        json.dump({"matrix": conf_matrix.tolist(), "classes": classes}, jf, indent=2)

    # --- 2. RUN GENERATOR LOSS SIMULATION ---
    print("\n" + "="*50)
    print("SIMULATING GENERATOR EMOTION LOSS (Loss_G_emo)")
    print("="*50)
    
    # Create a small batch of FAKE data and target the 'happy' label (index 0)
    # This tensor would be the output of your G.decoder(latent) in the real loop
    
    # NOTE: The values here are NOT real and are just for demonstration!
    fake_notes_tensor = torch.randn(4, 512, 4, device=device) 
    intended_labels = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device) # Intended: Happy
    
    g_emo_loss = calculate_generator_emotion_loss(fake_notes_tensor, intended_labels, ed_model)
    
    print(f"Simulated Loss_G_emo for 'Happy' target: {g_emo_loss:.4f}")