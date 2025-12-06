import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from tqdm.notebook import tqdm
import re

# --- Configuration (Relative) ---
ROOT_PATH = Path('dataset_root')
DOCS_PATH = ROOT_PATH / 'docs'
PROCESSED_PATH = ROOT_PATH / 'processed'
MODELS_PATH = ROOT_PATH / 'models'
SPLITS_PATH = ROOT_PATH / 'splits'

# 1. Paths
MANIFEST_FILE = DOCS_PATH / 'data_manifest.csv'
MODELS_PATH.mkdir(exist_ok=True)
SPLITS_PATH.mkdir(exist_ok=True)

# 2. Split Configuration
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# --- Helper Functions ---
def sanitize_filename(stem):
    """This MUST be identical to the one in your preprocessing script."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)

def get_npz_path(row):
    """Finds the .npz file from the manifest row."""
    original_stem = Path(row['full_path']).stem
    sanitized_stem = sanitize_filename(original_stem)
    filename = f"{row['source']}_{sanitized_stem}.npz"
    # Return a relative path
    return PROCESSED_PATH / filename

# --- Main Execution ---
print(f"Loading manifest from {MANIFEST_FILE}...")
manifest = pd.read_csv(MANIFEST_FILE)
manifest['npz_path'] = manifest.apply(get_npz_path, axis=1)
print(f"Found {len(manifest)} total labeled samples.")

# --- 1. Create Train/Val/Test Splits ---
train_df, temp_df = train_test_split(
    manifest,
    test_size=(TEST_SIZE + VAL_SIZE),
    random_state=RANDOM_STATE,
    stratify=manifest['emotion']
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_df['emotion']
)
print("\n--- Split Report ---")
print(f"Training samples:   {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples:     {len(test_df)}")

# --- 2. Fit Scaler on *Training Data Only* ---
print("\nFitting StandardScaler on training data...")
numeric_features_list = []
missing_files_count = 0

for npz_path in tqdm(train_df['npz_path'], desc="Loading numeric features"):
    if not npz_path.exists():
        # This will catch the 9 corrupt files, which is expected
        missing_files_count += 1
        continue
    try:
        data = np.load(npz_path)

        # --- THIS IS THE FIX ---
        # Look for 'numeric_features' (with an 's')
        numeric_features_list.append(data['numeric_features'])
        # ---------------------

    except KeyError:
        print(f"Warning: Could not load {npz_path}. Error: Key 'numeric_features' not found.")
        missing_files_count += 1
    except Exception as e:
        print(f"Warning: Could not load {npz_path}. Error: {e}")

if not numeric_features_list:
    print("\nFATAL ERROR: No numeric features were loaded.")
    sys.exit()

print(f"\nLoaded {len(numeric_features_list)} numeric features for scaling.")
if missing_files_count > 0:
    print(f"Skipped {missing_files_count} missing/corrupt files (this is expected).")
else:
    print("All training files found and loaded!")

all_train_numeric = np.vstack(numeric_features_list)
scaler = StandardScaler()
scaler.fit(all_train_numeric)
print("Scaler fitted successfully.")
print(f"  - Feature Means: {scaler.mean_}")
print(f"  - Feature Scales: {scaler.scale_}")

# --- 3. Save Outputs (as relative paths) ---
scaler_output_path = MODELS_PATH / 'scaler.joblib'
joblib.dump(scaler, scaler_output_path)
print(f"\nScaler saved to: {scaler_output_path}")

train_output_path = SPLITS_PATH / 'train_split.csv'
val_output_path = SPLITS_PATH / 'val_split.csv'
test_output_path = SPLITS_PATH / 'test_split.csv'

train_df.to_csv(train_output_path, index=False)
val_df.to_csv(val_output_path, index=False)
test_df.to_csv(test_output_path, index=False)
print(f"Train split saved to: {train_output_path}")
print(f"Val split saved to:   {val_output_path}")
print(f"Test split saved to:  {test_output_path}")