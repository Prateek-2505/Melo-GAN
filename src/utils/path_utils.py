# src/ae/utils.py
import os
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)