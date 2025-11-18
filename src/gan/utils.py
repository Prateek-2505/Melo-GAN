"""
Utilities for GAN training and inference.
Includes: Seeding, Weights Init, WGAN-GP Loss, and MIDI Post-Processing.
"""

import torch
import torch.autograd as autograd
import random
import numpy as np
import os
import pretty_midi # Requires: pip install pretty_midi

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
        if getattr(m, 'bias', None) is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

def load_ae_decoder_into_generator(ae_ckpt_path, generator):
    if not os.path.exists(ae_ckpt_path):
        print(f"[WARN] AE full checkpoint not found at {ae_ckpt_path}")
        return False
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    model_state = ckpt.get('model_state', None)
    if model_state is None:
        return False
    decoder_state = {k.replace('decoder.', ''): v for k, v in model_state.items() if k.startswith('decoder.')}
    gen_dec_state = generator.decoder.state_dict()
    loadable = {k: v for k, v in decoder_state.items() if k in gen_dec_state and gen_dec_state[k].shape == v.shape}
    gen_dec_state.update(loadable)
    generator.decoder.load_state_dict(gen_dec_state)
    print(f"[INFO] loaded {len(loadable)} decoder params from AE ckpt into generator.decoder")
    return True

def emotion_to_index(emotion):
    if emotion is None: return -1
    if isinstance(emotion, (list, tuple, np.ndarray)):
        arr = np.array(emotion)
        if arr.ndim == 1 and arr.size == 4: return int(np.argmax(arr))
        else: return int(arr)
    if isinstance(emotion, str):
        mapping = {'happy':0, 'sad':1, 'angry':2, 'calm':3}
        return mapping.get(emotion.lower(), -1)
    try: return int(emotion)
    except: return -1

# --- THIS IS THE CORRECTED FUNCTION ---
def compute_gradient_penalty(D, real_samples, fake_samples, numeric_embedding, device):
    """
    Calculates the gradient penalty loss for WGAN GP.
    'D' is the Discriminator (Critic).
    """
    # Random weight term for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Get Discriminator score for interpolated samples
    # --- FIX IS HERE ---
    # The Discriminator (D) now only returns one value (the score).
    d_interpolates = D(interpolates, numeric_embedding)

    # Get gradient w.r.t. the interpolated samples
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients
    # Use .reshape() instead of .view() to handle non-contiguous tensors
    gradients = gradients.reshape(gradients.size(0), -1)

    # Calculate penalty: (||grad||_2 - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ... (all other functions in utils.py like seed_everything, etc.)

# ... (all other functions in utils.py)

# --- NEW: Post-Processing (MIDI Generation) ---
# ... (in src/gan/utils.py)
# ... (keep all other functions like seed_everything, compute_gradient_penalty)

# --- NORMALIZATION CONSTANTS (Must match preprocessing) ---
MAX_BEAT_TIME = 4.0



def save_piano_roll_to_midi(notes_array, output_path, fs=100, bpm=120.0, scale_type='chromatic'):
    """
    Converts GAN output to MIDI with MUSICAL THEORY ENFORCEMENT.
    - scale_type: 'major', 'minor', or 'chromatic'
    """
    bpm = max(60, min(bpm, 180))
    seconds_per_beat = 60.0 / bpm

    piano_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano_inst = pretty_midi.Instrument(program=piano_program)

    current_time_beats = 0.0
    
    # --- 1. AGGRESSIVE SILENCE THRESHOLD ---
    # Increased from -0.4 to -0.2 to remove more "clutter" notes
    VELOCITY_THRESHOLD = -0.2 

    # --- 2. SCALE DEFINITIONS (C Major / C Minor) ---
    # Notes allowed in C Major: C, D, E, F, G, A, B
    SCALE_MAJOR = [0, 2, 4, 5, 7, 9, 11]
    # Notes allowed in C Minor: C, D, Eb, F, G, Ab, Bb
    SCALE_MINOR = [0, 2, 3, 5, 7, 8, 10]

    def snap_to_scale(pitch, scale_indices):
        """Finds the nearest valid MIDI note in the given scale."""
        octave = pitch // 12
        note_in_octave = pitch % 12
        # Find closest valid note
        closest = min(scale_indices, key=lambda x: abs(x - note_in_octave))
        return (octave * 12) + closest

    for note_info in notes_array:
        norm_pitch, norm_velocity, norm_duration, norm_step = note_info
        
        # Step size is always calculated to keep time moving
        step_beats = max(0.1, ((norm_step + 1.0) / 2.0) * MAX_BEAT_TIME)
        
        # CHECK SILENCE
        if norm_velocity < VELOCITY_THRESHOLD:
            current_time_beats += step_beats
            continue 

        # --- PITCH PROCESSING ---
        # 1. Un-normalize to MIDI range
        pitch = int(((norm_pitch + 1.0) * 63.5))
        pitch = np.clip(pitch, 36, 96) # Limit to reasonable piano range (C2-C7)
        
        # 2. Scale Snapping (The "Musicality" Fix)
        if scale_type == 'major':
            pitch = snap_to_scale(pitch, SCALE_MAJOR)
        elif scale_type == 'minor':
            pitch = snap_to_scale(pitch, SCALE_MINOR)
        
        # --- VELOCITY PROCESSING ---
        vel_range = 1.0 - VELOCITY_THRESHOLD
        vel_offset = norm_velocity - VELOCITY_THRESHOLD
        velocity = int(60 + (vel_offset / vel_range) * 67) # Min volume 60 (clearer sound)
        velocity = np.clip(velocity, 0, 127)
        
        # --- DURATION PROCESSING ---
        # Minimum duration 0.25 beats (16th note) to avoid "glitchy" short notes
        duration_beats = max(0.25, ((norm_duration + 1.0) / 2.0) * MAX_BEAT_TIME)
        
        # Add Note
        start_sec = current_time_beats * seconds_per_beat
        end_sec = (current_time_beats + duration_beats) * seconds_per_beat

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_sec,
            end=end_sec
        )
        piano_inst.notes.append(note)
        
        current_time_beats += step_beats

    piano_midi.instruments.append(piano_inst)
    piano_midi.write(output_path)
    
    final_time_sec = piano_midi.get_end_time()
    print(f"[INFO] Saved MIDI ({scale_type} scale) to {output_path}") 
    