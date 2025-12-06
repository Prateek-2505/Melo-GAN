import os
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import pretty_midi

# Import your project modules
from src.gan.models import Generator
from src.gan.feature_encoder import FeatureEncoder
from src.gan.utils import save_piano_roll_to_midi, seed_everything

# --- Configuration ---
GAN_CONFIG_PATH = "config/gan_config.yaml"
GAN_CHECKPOINT = "experiments/gan/checkpoints/gan_final.pth"
TEMP_MIDI_DIR = "eval_temp_midi"

# --- TARGET METRICS (Based on your Training Goals) ---
TARGET_STATS = {
    'Happy': {'pitch': 73.0, 'velocity': 85.0, 'density': 8.0},
    'Sad':   {'pitch': 54.0, 'velocity': 60.0, 'density': 2.0},
    'Angry': {'pitch': 75.0, 'velocity': 90.0, 'density': 10.0},
    'Calm':  {'pitch': 54.0, 'velocity': 58.0, 'density': 2.0}
}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --- INSTRUMENT & MUSICAL LOGIC ---
INSTRUMENTS = {
    'happy': ['Acoustic Grand Piano', 'Bright Acoustic Piano', 'Clavinet', 'Marimba', 'Steel Drums', 'Celesta'],
    'sad': ['Acoustic Grand Piano', 'Electric Piano 2', 'Church Organ', 'Reed Organ', 'Acoustic Guitar (nylon)'],
    'angry': ['Acoustic Grand Piano', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Tango Accordion'],
    'calm': ['Acoustic Grand Piano', 'Electric Piano 1', 'Vibraphone', 'Music Box', 'Acoustic Guitar (steel)']
}
COMMON_ROOTS = [0, 2, 4, 5, 7, 9, 10] 

def get_musical_params(emotion):
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

def evaluate_gan(device):
    print("\n" + "="*40)
    print("      EVALUATING GAN (MSE METRICS)")
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
    
    emotion_vectors = {
        "Happy": [1.0, 1.0, 0.8, 0.8, 0.5, 0.5],
        "Sad":   [-1.0, -1.0, -0.5, -0.5, -0.5, -0.5],
        "Angry": [1.0, -1.0, 1.0, 1.0, -0.8, 0.8],
        "Calm":  [-1.0, 1.0, -0.8, -0.8, 0.5, -0.5]
    }
    
    num_samples = 50 
    
    mse_report = {e: {"pitch": 0.0, "velocity": 0.0, "density": 0.0} for e in emotions}
    raw_stats = {e: {"pitch": [], "velocity": [], "density": []} for e in emotions}

    print(f"[INFO] Analyzing {num_samples} samples per emotion (Skipping generation if files exist)...")

    for emo_name in emotions:
        # We still run the generator to keep the batch logic consistent, 
        # but we won't save the output if files exist.
        base_feat = torch.tensor([emotion_vectors[emo_name]] * num_samples).float().to(device)
        jitter = torch.randn_like(base_feat) * 0.1
        numeric_emb = E_num(base_feat + jitter)
        
        noise = torch.randn(num_samples, cfg['NOISE_DIM']).to(device)
        encoder_latent = torch.zeros(num_samples, cfg['LATENT_DIM']).to(device)

        with torch.no_grad():
            gen_notes, _ = G(noise, encoder_latent, numeric_emb)
            gen_notes_cpu = gen_notes.cpu().numpy()

            for i in range(num_samples):
                fpath = os.path.join(TEMP_MIDI_DIR, f"eval_{emo_name}_{i}.mid")
                
                # --- CHECK: Only generate if file is missing ---
                if not os.path.exists(fpath):
                    root, scale, bpm, instrument = get_musical_params(emo_name)
                    save_piano_roll_to_midi(gen_notes_cpu[i], fpath, instrument_name=instrument, bpm=bpm, scale=scale, root_key=root)
                
                # --- ANALYZE: Always read from disk ---
                stats = analyze_generated_midi(fpath)
                if stats:
                    raw_stats[emo_name]["pitch"].append(stats["avg_pitch"])
                    raw_stats[emo_name]["velocity"].append(stats["avg_velocity"])
                    raw_stats[emo_name]["density"].append(stats["density"])
                    
                    target = TARGET_STATS[emo_name]
                    mse_report[emo_name]["pitch"] += (stats["avg_pitch"] - target['pitch']) ** 2
                    mse_report[emo_name]["velocity"] += (stats["avg_velocity"] - target['velocity']) ** 2
                    mse_report[emo_name]["density"] += (stats["density"] - target['density']) ** 2

    # --- Final Calculations & Report ---
    print("\n" + "-"*65)
    print("FEATURE ERROR ANALYSIS (MSE - Lower is Better)")
    print("-" * 65)
    print(f"{'Emotion':<10} | {'Pitch MSE':<12} | {'Vel MSE':<12} | {'Density MSE':<12}")
    print("-" * 65)

    for emo in emotions:
        count = len(raw_stats[emo]["pitch"])
        if count > 0:
            mse_p = mse_report[emo]["pitch"] / count
            mse_v = mse_report[emo]["velocity"] / count
            mse_d = mse_report[emo]["density"] / count
            print(f"{emo:<10} | {mse_p:<12.2f} | {mse_v:<12.2f} | {mse_d:<12.2f}")
        else:
            print(f"{emo:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")

    print("\n" + "-"*65)
    print("RAW GENERATED STATS (Average)")
    print("-" * 65)
    print(f"{'Emotion':<10} | {'Pitch':<12} | {'Velocity':<12} | {'Density':<12}")
    print("-" * 65)
    
    for emo in emotions:
        count = len(raw_stats[emo]["pitch"])
        if count > 0:
            avg_p = np.mean(raw_stats[emo]["pitch"])
            avg_v = np.mean(raw_stats[emo]["velocity"])
            avg_d = np.mean(raw_stats[emo]["density"])
            print(f"{emo:<10} | {avg_p:<12.1f} | {avg_v:<12.1f} | {avg_d:<12.2f}")
        else:
            print(f"{emo:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12}")

    print("-" * 65)
    print(f"Temporary MIDI files saved in: {TEMP_MIDI_DIR}")

if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on: {device}")
    evaluate_gan(device)