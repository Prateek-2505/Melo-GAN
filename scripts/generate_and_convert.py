#!/usr/bin/env python3
"""
Attempt to load a Generator from common module paths and a checkpoint,
sample, save .npy and write simple MIDI (pretty_midi required).
Drop-in diagnostic tool for MELO-GAN-style projects.
"""
import os
import sys
import argparse
import importlib
from glob import glob
import numpy as np

try:
    import torch
    import pretty_midi
except Exception as e:
    print("ERROR: required packages missing. Install with:\n  pip install torch pretty_midi numpy")
    raise

CANDIDATE_GENERATOR_MODULES = [
    "src.models.generator",
    "src.models.gan.generator",
    "src.gan.generator",
    "models.generator",
    "gan.generator",
    "src.models.gan.model",
    "src.gan.model",
    "models.gan.generator",
]

CANDIDATE_CHECKPOINTS = [
    "data/models/generator_best.pth",
    "data/models/generator.pt",
    "data/models/gan/generator_best.pth",
    "data/models/gan_generator.pth",
    "checkpoints/generator_best.pth",
    "checkpoints/g_*.pth",
    "models/generator_best.pth",
]

def try_import_generator():
    for mod in CANDIDATE_GENERATOR_MODULES:
        try:
            m = importlib.import_module(mod)
            # try common class names
            for cls_name in ("Generator", "GANGenerator", "NetG", "GeneratorNet"):
                if hasattr(m, cls_name):
                    print(f"Found generator class {cls_name} in module {mod}")
                    return getattr(m, cls_name), mod
            # maybe module exposes create_generator or build_generator
            for fn in ("create_generator", "build_generator", "get_generator"):
                if hasattr(m, fn):
                    print(f"Found factory {fn} in module {mod}")
                    return getattr(m, fn), mod
        except Exception:
            continue
    return None, None

def find_checkpoint():
    for cp in CANDIDATE_CHECKPOINTS:
        if "*" in cp:
            matches = glob(cp)
            if matches:
                return matches[-1]
            continue
        if os.path.exists(cp):
            return cp
    # fallback: any .pth or .pt in data/models
    for p in glob("data/models/**/*.*", recursive=True):
        if p.endswith((".pth", ".pt", ".ckpt")):
            return p
    return None

def basic_convert_array_to_midi(arr, outpath, tempo=120):
    """
    arr: numpy array shape (seq_len, features) or (seq_len,)
    This is a heuristic: if arr is continuous, convert values to pitches in a range
    """
    seq = arr
    if seq.ndim > 1:
        # pick first column if multi-feature
        seq = seq[:,0]
    # normalize to midi pitch 48-84
    mn, mx = seq.min(), seq.max()
    if mx - mn < 1e-6:
        pitches = np.clip(np.round(60 + np.zeros_like(seq)), 0, 127).astype(int)
    else:
        pitches = 48 + ((seq - mn)/(mx-mn) * 36).astype(int)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)  # acoustic grand piano
    time = 0.0
    dur = 0.25  # quarter-note default
    for p in pitches:
        note = pretty_midi.Note(velocity=100, pitch=int(p), start=time, end=time+dur)
        inst.notes.append(note)
        time += dur
    pm.instruments.append(inst)
    pm.write(outpath)
    print("Wrote midi to", outpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="generated_samples")
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    GenClass, mod = try_import_generator()
    cp = args.checkpoint or find_checkpoint()
    print("Selected checkpoint:", cp)
    if GenClass is None:
        print("Warning: Could not find a Generator class in common modules.")
        print("If you know the generator module/class, set --checkpoint and adjust the script.")
    else:
        print(f"Attempting to instantiate generator from {mod} ...")

    if cp is None:
        print("No checkpoint found in candidates.")
    else:
        try:
            ck = torch.load(cp, map_location="cpu")
            # if checkpoint is dict with state_dict
            if isinstance(ck, dict):
                sd = None
                if "state_dict" in ck:
                    sd = ck["state_dict"]
                elif "generator_state_dict" in ck:
                    sd = ck["generator_state_dict"]
                else:
                    # maybe it's a plain state dict
                    sd = ck
            else:
                sd = ck
            print("Checkpoint loaded. Keys preview:", list(sd.keys())[:10] if isinstance(sd, dict) else "non-dict")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            sd = None

        # try to instantiate generator and load state dict
        if GenClass is not None and sd is not None:
            try:
                # attempt zero-arg or latent-dim init
                try:
                    G = GenClass()
                except TypeError:
                    try:
                        G = GenClass(latent_dim=args.latent_dim)
                    except Exception:
                        G = GenClass(*[])
                G.load_state_dict(sd)
                G.eval()
                print("Generator instantiated and weights loaded.")
                device = torch.device("cpu")
                G.to(device)
                for i in range(args.n_samples):
                    z = torch.randn(1, args.latent_dim, device=device)
                    with torch.no_grad():
                        out = G(z)
                    arr = out.cpu().numpy().squeeze()
                    npy_path = os.path.join(args.outdir, f"sample_{i}.npy")
                    np.save(npy_path, arr)
                    midi_path = os.path.join(args.outdir, f"sample_{i}.mid")
                    basic_convert_array_to_midi(arr, midi_path)
                print("Done sampling using loaded generator.")
                return
            except Exception as e:
                print("Generator instantiation / loading failed:", e)

    # fallback: try any 'sample' function in repo
    print("Fallback: searching for functions named 'sample' in the repo...")
    # naive fallback: try to find .npy generation utils
    candidates = glob("**/*sample*.py", recursive=True)[:20]
    print("sample-related files found (first 20):", candidates)
    print("If nothing generated, implement a project-specific sampler or tell me exact module/class names.")

if __name__ == "__main__":
    main()
