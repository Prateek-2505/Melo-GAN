import os
import yaml
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import pretty_midi
import sys
import os

# --- FIX: Add Project Root to Path ---
# This allows the script to see 'src' even when inside 'graph_plotting'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------
# Import your project modules
from src.emotion_discriminator.ed_model import EmotionDiscriminator
from src.emotion_discriminator.ed_dataset import build_dataloader
from src.gan.models import Generator
from src.gan.feature_encoder import FeatureEncoder
from src.gan.utils import save_piano_roll_to_midi, seed_everything

# --- Configuration ---
ED_CONFIG_PATH = "config/ed_config.yaml"
GAN_CONFIG_PATH = "config/gan_config.yaml"
ED_CHECKPOINT = "data/models/ed/ed_best.pth"
GAN_CHECKPOINT = "experiments/gan/checkpoints/gan_final.pth"
TEMP_MIDI_DIR = "eval_temp_midi"
PLOTS_DIR = "eval_plots"  # Directory to save graphs

# Path to training history (created during ED training)
HISTORY_PATH = "experiments/ed/logs/ed_training_history.json"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --- INSTRUMENT & MUSICAL LOGIC ---
INSTRUMENTS = {
    'happy': ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Piano 1', 'Clavinet', 'Marimba', 'Steel Drums', 'Celesta'],
    'sad': ['Acoustic Grand Piano', 'Electric Piano 2', 'Church Organ', 'Reed Organ', 'Acoustic Guitar (nylon)', 'Harp'],
    'angry': ['Acoustic Grand Piano', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Harpsichord', 'Orchestral Harp', 'Tango Accordion'],
    'calm': ['Acoustic Grand Piano', 'Electric Piano 1', 'Vibraphone', 'Music Box', 'Acoustic Guitar (steel)', 'Glockenspiel']
}
COMMON_ROOTS = [0, 2, 4, 5, 7, 9, 10] 


def get_musical_params(emotion):
    """Returns a random (Root Key, Scale, BPM, Instrument) based on emotion."""
    emotion = emotion.lower()
    root = random.choice(COMMON_ROOTS)
    
    possible_instruments = INSTRUMENTS.get(emotion, ['Acoustic Grand Piano'])
    instrument = random.choice(possible_instruments)

    if emotion == "happy":
        scale = random.choice(['major', 'lydian', 'mixolydian', 'major_pentatonic'])
        bpm = random.randint(120, 150)
    elif emotion == "sad":
        scale = random.choice(['minor', 'dorian', 'phrygian'])
        bpm = random.randint(60, 85)
    elif emotion == "angry":
        scale = random.choice(['minor', 'minor_pentatonic', 'blues', 'locrian'])
        bpm = random.randint(140, 170)
    elif emotion == "calm":
        scale = random.choice(['major_pentatonic', 'lydian', 'major'])
        bpm = random.randint(70, 95)
    else:
        scale = 'major'
        bpm = 120
        instrument = 'Acoustic Grand Piano'
        
    return root, scale, bpm, instrument


# --- PLOTTING FUNCTIONS ---
def plot_ed_training_metrics(history, save_dir):
    """Plots Loss and Accuracy over epochs from history dict."""
    print("[INFO] Plotting Training Curves...")
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss Axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss', color=color)
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color=color, linestyle='-')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Val Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Accuracy Axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  
    ax2.plot(epochs, history['train_acc'], label='Train Acc', color=color, linestyle='-')
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], label='Val Acc', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Emotion Discriminator Training Dynamics')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ed_training_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Training metrics saved to {save_path}")


def plot_confusion_matrix(cm, classes, save_dir):
    """Plots a heatmap of True vs Predicted emotions."""
    print("[INFO] Plotting Confusion Matrix...")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Emotion Discriminator Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ed_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


# --- 1. Emotion Discriminator Evaluation ---
def evaluate_emotion_discriminator(device):
    print("\n" + "="*40)
    print("   EVALUATING EMOTION DISCRIMINATOR")
    print("="*40)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load Config & Model
    cfg = load_config(ED_CONFIG_PATH)
    model = EmotionDiscriminator(cfg).to(device)
    
    if not os.path.exists(ED_CHECKPOINT):
        print(f"[ERROR] ED Checkpoint not found at {ED_CHECKPOINT}")
        return
    
    ckpt = torch.load(ED_CHECKPOINT, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Load Test Data
    try:
        test_loader = build_dataloader(cfg, split="test", shuffle=False)
        print(f"[INFO] Loaded Test Set: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"[WARN] Could not load test set ({e}), falling back to Validation set.")
        test_loader = build_dataloader(cfg, split="val", shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating ED"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            logits = model(x)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # --- Metrics & Text Report ---
    target_names = ["Happy", "Sad", "Angry", "Calm"]
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names)
    
    print(f"\n[RESULT] ED Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # --- 1. Generate & Plot Confusion Matrix (Live Data) ---
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, target_names, PLOTS_DIR)

    # --- 2. Plot Training Curves (From History File) ---
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
            plot_ed_training_metrics(history, PLOTS_DIR)
        except Exception as e:
            print(f"[WARN] Could not plot training metrics: {e}")
    else:
        print(f"[WARN] History file not found at {HISTORY_PATH}. Skipping training curve plot.")
    
    return model


# --- 2. GAN Evaluation ---
def analyze_generated_midi(filepath):
    try:
        pm = pretty_midi.PrettyMIDI(filepath)
        if len(pm.instruments) == 0 or len(pm.instruments[0].notes) == 0:
            return None
        
        notes = pm.instruments[0].notes
        pitches = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])
        duration = pm.get_end_time()
        
        return {
            "avg_pitch": np.mean(pitches),
            "avg_velocity": np.mean(velocities),
            "density": len(notes) / duration if duration > 0 else 0
        }
    except:
        return None

def evaluate_gan(ed_model, device):
    print("\n" + "="*40)
    print("        EVALUATING GAN PERFORMANCE")
    print("="*40)
    
    cfg = load_config(GAN_CONFIG_PATH)
    os.makedirs(TEMP_MIDI_DIR, exist_ok=True)

    numeric_dim = cfg.get('NUMERIC_INPUT_DIM', 6)
    embed_dim = cfg.get('ENCODER_OUT_DIM', 128)
    
    E_num = FeatureEncoder(in_dim=numeric_dim, out_dim=embed_dim).to(device)
    G = Generator(
        noise_dim=cfg['NOISE_DIM'], 
        latent_dim=cfg['LATENT_DIM'], 
        mode=cfg.get('INTEGRATION_MODE', 'warm_start'), 
        max_notes=cfg['MAX_NOTES'], 
        note_dim=cfg['NOTE_DIM'],
        numeric_embed_dim=embed_dim
    ).to(device)

    if not os.path.exists(GAN_CHECKPOINT):
        print(f"[ERROR] GAN Checkpoint not found at {GAN_CHECKPOINT}")
        return

    ckpt = torch.load(GAN_CHECKPOINT, map_location=device)
    G.load_state_dict(ckpt['G'])
    E_num.load_state_dict(ckpt['E_num'])
    G.eval()
    E_num.eval()

    emotions = ["Happy", "Sad", "Angry", "Calm"]
    
    # Ideal feature vectors (Normalized -1 to 1)
    emotion_vectors = {
        "Happy": [1.0, 1.0, 0.8, 0.8, 0.5, 0.5],
        "Sad":   [-1.0, -1.0, -0.5, -0.5, -0.5, -0.5],
        "Angry": [1.0, -1.0, 1.0, 1.0, -0.8, 0.8],
        "Calm":  [-1.0, 1.0, -0.8, -0.8, 0.5, -0.5]
    }
    
    num_samples = 50
    consistency_scores = {}
    musical_stats = {e: {"pitch": [], "density": [], "velocity": []} for e in emotions}

    print(f"[INFO] Generating {num_samples} samples per emotion...")

    for emo_name in emotions:
        target_idx = emotions.index(emo_name)
        base_feat = torch.tensor([emotion_vectors[emo_name]] * num_samples).float().to(device)
        
        jitter = torch.randn_like(base_feat) * 0.1
        numeric_emb = E_num(base_feat + jitter)
        
        noise = torch.randn(num_samples, cfg['NOISE_DIM']).to(device)
        encoder_latent = torch.zeros(num_samples, cfg['LATENT_DIM']).to(device)

        with torch.no_grad():
            gen_notes, _ = G(noise, encoder_latent, numeric_emb)
            
            # 1. Check with Emotion Discriminator
            ed_logits = ed_model(gen_notes)
            ed_preds = ed_logits.argmax(dim=1).cpu().numpy()
            
            correct = (ed_preds == target_idx).sum()
            accuracy = correct / num_samples
            consistency_scores[emo_name] = accuracy

            # 2. Analyze Musical Features
            gen_notes_cpu = gen_notes.cpu().numpy()
            for i in range(num_samples):
                fpath = os.path.join(TEMP_MIDI_DIR, f"eval_{emo_name}_{i}.mid")
                root, scale, bpm, instrument = get_musical_params(emo_name)
                save_piano_roll_to_midi(gen_notes_cpu[i], fpath, instrument_name=instrument,bpm=bpm,scale=scale,root_key=root)
                
                stats = analyze_generated_midi(fpath)
                if stats:
                    musical_stats[emo_name]["pitch"].append(stats["avg_pitch"])
                    musical_stats[emo_name]["density"].append(stats["density"])
                    musical_stats[emo_name]["velocity"].append(stats["avg_velocity"])

    # --- Print GAN Report ---
    print("\n" + "-"*40)
    print("GAN EVALUATION REPORT")
    print("-"*40)
    
    print("\n1. Emotional Consistency (Does the GAN fool the ED?)")
    avg_consistency = 0
    for emo, score in consistency_scores.items():
        print(f"   - {emo}: {score:.2f}")
        avg_consistency += score
    print(f"   > AVERAGE: {avg_consistency/4:.2f}")

    print("\n2. Musical Feature Analysis")
    print(f"   {'Emotion':<10} | {'Avg Pitch':<10} | {'Density (n/s)':<15} | {'Avg Velocity':<10}")
    print("   " + "-"*55)
    
    for emo in emotions:
        p = np.mean(musical_stats[emo]["pitch"]) if musical_stats[emo]["pitch"] else 0
        d = np.mean(musical_stats[emo]["density"]) if musical_stats[emo]["density"] else 0
        v = np.mean(musical_stats[emo]["velocity"]) if musical_stats[emo]["velocity"] else 0
        print(f"   {emo:<10} | {p:<10.1f} | {d:<15.2f} | {v:<10.1f}")

    print("-" * 40)
    print(f"Temporary MIDI files saved in: {TEMP_MIDI_DIR}")

if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on: {device}")
    
    # Run ED Evaluation (This now generates PLOTS)
    trained_ed = evaluate_emotion_discriminator(device)
    
    # Run GAN Evaluation
    if trained_ed:
        evaluate_gan(trained_ed, device)