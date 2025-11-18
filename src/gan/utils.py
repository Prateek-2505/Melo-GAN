"""
Utilities for GAN training and inference.
Includes: Training Utils, WGAN Loss, Advanced Music Theory (Scales), and Instrument Support.
"""

import torch
import torch.autograd as autograd
import random
import numpy as np
import os
import pretty_midi

# --- 1. MUSICAL SCALE DEFINITIONS ---
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'major_pentatonic': [0, 2, 4, 7, 9],
    'minor_pentatonic': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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

def compute_gradient_penalty(D, real_samples, fake_samples, numeric_embedding, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, numeric_embedding)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# --- MIDI GENERATION ---
MAX_BEAT_TIME = 4.0

def save_piano_roll_to_midi(notes_array, output_path, fs=100, bpm=120.0, scale='major', root_key=0, instrument_name='Acoustic Grand Piano'):
    """
    Converts GAN output to MIDI with:
    1. Scale Snapping (Key enforcement)
    2. Dynamic BPM
    3. Dynamic Instrument Selection
    """
    bpm = max(60, min(bpm, 180))
    seconds_per_beat = 60.0 / bpm

    piano_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    
    # --- INSTRUMENT SELECTION ---
    try:
        program = pretty_midi.instrument_name_to_program(instrument_name)
    except:
        print(f"[WARN] Instrument '{instrument_name}' not found. Defaulting to Piano.")
        program = 0 
        
    piano_inst = pretty_midi.Instrument(program=program)

    current_time_beats = 0.0
    VELOCITY_THRESHOLD = -0.2 

    # Build Allowed Notes
    intervals = SCALES.get(scale, SCALES['chromatic'])
    allowed_notes = [(interval + root_key) % 12 for interval in intervals]
    allowed_notes.sort()

    def snap_to_scale(pitch):
        octave = pitch // 12
        note_in_octave = pitch % 12
        closest = min(allowed_notes, key=lambda x: abs(x - note_in_octave))
        return (octave * 12) + closest

    for note_info in notes_array:
        norm_pitch, norm_velocity, norm_duration, norm_step = note_info
        
        step_beats = max(0.1, ((norm_step + 1.0) / 2.0) * MAX_BEAT_TIME)
        
        if norm_velocity < VELOCITY_THRESHOLD:
            current_time_beats += step_beats
            continue 

        pitch = int(((norm_pitch + 1.0) * 63.5))
        pitch = np.clip(pitch, 36, 96) 
        pitch = snap_to_scale(pitch)
        
        vel_range = 1.0 - VELOCITY_THRESHOLD
        vel_offset = norm_velocity - VELOCITY_THRESHOLD
        velocity = int(60 + (vel_offset / vel_range) * 67)
        velocity = np.clip(velocity, 0, 127)
        
        duration_beats = max(0.25, ((norm_duration + 1.0) / 2.0) * MAX_BEAT_TIME)
        
        start_sec = current_time_beats * seconds_per_beat
        end_sec = (current_time_beats + duration_beats) * seconds_per_beat

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_sec, end=end_sec)
        piano_inst.notes.append(note)
        current_time_beats += step_beats

    piano_midi.instruments.append(piano_inst)
    piano_midi.write(output_path)
    
    scale_name = f"{NOTE_NAMES[root_key]} {scale}"
    print(f"[INFO] Saved MIDI ({instrument_name} | {scale_name}) to {output_path}")