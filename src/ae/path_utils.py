# src/ae/path_utils.py
import os
import yaml

def load_config(path):
    """
    Load YAML config. Accepts absolute or relative paths.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
