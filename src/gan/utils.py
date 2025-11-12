"""
Utilities for GAN training: seeding, weight init, loading AE checkpoints into generator (warm_start fallback),
and helper to convert one-hot emotion to index.
"""

import torch
import random
import numpy as np
import os

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
    """
    Try to load decoder weights from AE checkpoint into the generator.decoder.
    ae_ckpt_path: path to full AE checkpoint (saved by train_ae.py as 'ae_epochXX_valYYY.pth' containing 'model_state' for full model)
    generator: Generator instance with .decoder attribute compatible with AE ConvDecoder parameter names
    Returns True if successful.
    """
    if not os.path.exists(ae_ckpt_path):
        print(f"[WARN] AE full checkpoint not found at {ae_ckpt_path}")
        return False
    ckpt = torch.load(ae_ckpt_path, map_location='cpu')
    # attempt to find decoder state in ckpt: train_ae saved full checkpoint with keys: 'model_state'
    model_state = ckpt.get('model_state', None)
    if model_state is None:
        print("[WARN] checkpoint missing 'model_state'. Cannot load decoder weights.")
        return False
    # build a mapping by matching keys containing 'decoder' from AE to generator.decoder
    decoder_state = {k.replace('decoder.', ''): v for k, v in model_state.items() if k.startswith('decoder.')}
    if not decoder_state:
        print("[WARN] no decoder.* keys found in AE checkpoint")
        return False
    # now get generator.decoder state_dict and attempt to load matching keys
    gen_dec_state = generator.decoder.state_dict()
    # filter decoder_state to keys present in gen_dec_state
    loadable = {}
    for k, v in decoder_state.items():
        if k in gen_dec_state and gen_dec_state[k].shape == v.shape:
            loadable[k] = v
    if not loadable:
        print("[WARN] no matching keys between AE decoder and generator.decoder")
        return False
    gen_dec_state.update(loadable)
    generator.decoder.load_state_dict(gen_dec_state)
    print(f"[INFO] loaded {len(loadable)} decoder params from AE ckpt into generator.decoder")
    return True

def emotion_to_index(emotion):
    """
    Accepts:
    - numeric one-hot arrays (length 4)
    - integer indices
    - strings ['happy','sad','angry','calm']
    Returns index 0..3
    """
    if emotion is None:
        return -1
    if isinstance(emotion, (list, tuple, np.ndarray)):
        arr = np.array(emotion)
        if arr.ndim == 1 and arr.size == 4:
            return int(np.argmax(arr))
        else:
            return int(arr)
    if isinstance(emotion, str):
        mapping = {'happy':0,'sad':1,'angry':2,'calm':3}
        return mapping.get(emotion.lower(), -1)
    try:
        return int(emotion)
    except:
        return -1
