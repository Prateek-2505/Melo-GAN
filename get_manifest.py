import pandas as pd
from pathlib import Path
import sys
import re # Import regular expressions

# --- Configuration (Now Relative) ---
ROOT_PATH = Path('dataset_root') # Relative to our current directory
DOCS_PATH = ROOT_PATH / 'docs'
RAW_PATH = ROOT_PATH / 'raw'

ORIGINAL_LABELS_CSV = DOCS_PATH / 'combined_standardized_labels.csv'

def get_file_details(file_key):
    """
    Based on the file_key, determine the source
    and the full path to the raw MIDI file.
    """
    file_key = str(file_key)
    path = None

    # Check for EMOPIA keys
    if file_key.startswith(('Q1_', 'Q2_', 'Q3_', 'Q4_')):
        source = 'emopia'
        path = RAW_PATH / 'labelled' / 'emopia' / f"{file_key}.mid"

    # Treat everything else as VGMIDI
    else:
        source = 'vgmidi'
        # Get filename from CSV (e.g., "Banjo..._0.mid")
        filename_from_csv = Path(file_key).name
        # Remove suffix (e.g., "_0.mid") and add ".mid" back
        final_filename = re.sub(r'_\d+\.mid$', '.mid', filename_from_csv)
        # Build the relative path
        path = RAW_PATH / 'labelled' / 'vgmidi' / final_filename

    return source, path

# --- Main Audit ---

print(f"Loading original labels from {ORIGINAL_LABELS_CSV}...")
try:
    df = pd.read_csv(ORIGINAL_LABELS_CSV)
except FileNotFoundError:
    print(f"\nFATAL ERROR: File not found at {ORIGINAL_LABELS_CSV}")
    print("Please check your file path and name.")
    sys.exit()
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    sys.exit()

print("Applying source and path mapping...")
df[['source', 'full_path']] = df.apply(
    lambda row: pd.Series(get_file_details(row['file_key'])),
    axis=1
)

print("Validating file existence...")
df['file_exists'] = df.apply(
    lambda row: Path(row['full_path']).exists() if row['full_path'] else False,
    axis=1
)

total_rows = len(df)
missing_files = df[~df['file_exists']]
valid_df = df[df['file_exists']].copy() # Create copy to avoid warning

print(f"\n--- Audit Report ---")
print(f"Total rows in CSV: {total_rows}")
print(f"Valid, existing files found: {len(valid_df)}")
print(f"Missing/Skipped files: {len(missing_files)}")

if not missing_files.empty:
    print("\nMissing files (first 5):")
    for _, row in missing_files.head().iterrows():
        print(f"  - Key: {row['file_key']}, Path: {row['full_path']}")
else:
    print("\n🎉 All 1282 files found successfully!")

# Standardize emotion strings
valid_df.loc[:, 'emotion'] = valid_df['emotion'].str.lower().str.strip()

# --- Save Manifest ---
output_manifest = DOCS_PATH / 'data_manifest.csv'
final_columns = ['file_key', 'emotion', 'source', 'full_path']
valid_df[final_columns].to_csv(output_manifest, index=False)

print(f"\nSuccess! Clean manifest saved to: {output_manifest}")