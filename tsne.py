import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import warnings

# --- Configuration ---
# ⚠️ You may need to change these paths to match your project
dir = 'train'
CSV_PATH = 'data/splits/'+dir+'_split.csv'
LATENTS_PATH = 'data/splits/'+dir+'/encoder_feats.npy'

# ℹ️ Based on ed_dataset.py
DEFAULT_LABELS = ["happy", "sad", "angry", "calm"]
LABEL_MAP = {label: i for i, label in enumerate(DEFAULT_LABELS)}
LABEL_COL = 'emotion' # Column in your CSV with labels
FILE_COL = 'npz_path'  # Column in your CSV with file IDs
# ---------------------

def load_and_align_data(csv_path, latents_path):
    """
    Loads latents and labels, aligning them based on the logic
    from ed_dataset.py.
    """
    print(f"Loading CSV from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return None, None
        
    df = pd.read_csv(csv_path)
    
    print(f"Loading latents from: {latents_path}")
    if not os.path.exists(latents_path):
        print(f"Error: Latents file not found at {latents_path}", file=sys.stderr)
        return None, None
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        latents_data = np.load(latents_path, allow_pickle=True)

    if LABEL_COL not in df.columns:
        print(f"Error: Label column '{LABEL_COL}' not found in CSV.", file=sys.stderr)
        return None, None
    if FILE_COL not in df.columns:
        print(f"Error: File ID column '{FILE_COL}' not found in CSV.", file=sys.stderr)
        return None, None

    latents_array = None
    labels_list = []
    
    # Case 1: Latents file is a dictionary-like object array
    # This logic mimics build_dataloader filtering
    if latents_data.dtype == np.object_:
        print("Latents file is an object array (map/dict). Aligning by key...")
        try:
            latents_map = dict(latents_data.tolist())
        except Exception as e:
            print(f"Error: Could not convert object array to dict: {e}", file=sys.stderr)
            return None, None
            
        def has_key(val):
            if pd.isna(val): return False
            s = str(val)
            return (s in latents_map) or (os.path.basename(s) in latents_map)
            
        mask = df[FILE_COL].apply(has_key)
        df_filtered = df[mask]
        
        if len(df_filtered) == 0:
            print("Error: No matching keys found between CSV and latents map.", file=sys.stderr)
            return None, None
            
        labels_list = df_filtered[LABEL_COL].tolist()
        keys_filtered = df_filtered[FILE_COL].tolist()
        
        aligned_latents_list = []
        for k in keys_filtered:
            s_k = str(k)
            vec = latents_map.get(s_k) or latents_map.get(os.path.basename(s_k))
            aligned_latents_list.append(vec)
            
        latents_array = np.array(aligned_latents_list, dtype=np.float32)
        
    # Case 2: Latents file is a standard NxD array
    # This logic mimics build_dataloader filtering
    else:
        print("Latents file is a standard array. Aligning by row order...")
        latents_array = np.asarray(latents_data, dtype=np.float32)
        arr_len = latents_array.shape[0]
        csv_len = len(df)
        
        if arr_len < csv_len:
            print(f"Warning: Latents array (len {arr_len}) is shorter than CSV (len {csv_len}). Truncating CSV.")
            df_filtered = df.iloc[:arr_len]
        else:
            df_filtered = df
            if arr_len > csv_len:
                print(f"Warning: Latents array (len {arr_len}) is longer than CSV (len {csv_len}). Truncating latents.")
                latents_array = latents_array[:csv_len]
                
        labels_list = df_filtered[LABEL_COL].tolist()

    if latents_array is None or len(labels_list) == 0:
        print("Error: Failed to load or align data.", file=sys.stderr)
        return None, None
        
    # Ensure latents are 2D
    if latents_array.ndim != 2:
        print(f"Error: Latents array is not 2D. Shape is {latents_array.shape}", file=sys.stderr)
        try:
            # Handle case where it might be (N, 1, D)
            latents_array = latents_array.reshape(latents_array.shape[0], -1)
            print(f"Reshaped latents to: {latents_array.shape}")
        except Exception as e:
            print(f"Could not reshape latents: {e}", file=sys.stderr)
            return None, None

    print(f"Successfully aligned {len(labels_list)} samples.")
    print(f"Final latents shape: {latents_array.shape}")
    
    return latents_array, labels_list

def plot_tsne(latents, labels):
    """
    Runs t-SNE and saves a colored scatter plot.
    """
    if latents is None or labels is None:
        print("Skipping plotting due to data loading errors.")
        return

    # Convert string labels to integers for coloring
    int_labels = [LABEL_MAP.get(str(l).lower().strip(), len(LABEL_MAP)) for l in labels]
    legend_names = list(LABEL_MAP.keys()) + ["Other"]
    
    print(f"\nRunning t-SNE on {latents.shape[0]} samples (dim={latents.shape[1]})...")
    print("(This may take a minute or two)")
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # A good default, can be tuned (5-50)
        n_iter=1000,
        random_state=42,
        n_jobs=-1       # Use all cores
    )
    tsne_results = tsne.fit_transform(latents)
    
    print("Plotting results...")
    
    # Create the scatter plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=int_labels,
        cmap='Spectral', # 'viridis' or 'Spectral' are good
        alpha=0.7,
        s=10 # point size
    )
    
    plt.title('t-SNE Visualization of VAE Latent Vectors '+dir, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create a legend
    try:
        handles, _ = scatter.legend_elements()
        # Filter legend to only show classes actually present
        present_classes_idx = sorted(list(set(int_labels)))
        present_handles = [handles[i] for i in present_classes_idx]
        present_names = [legend_names[i] for i in present_classes_idx]
        
        plt.legend(handles=present_handles, labels=present_names, title="Emotions")
    except Exception as e:
        print(f"Could not create full legend: {e}")
        plt.legend()

    
    # Save the figure
    output_filename = dir+'_latent_tsne_visualization.png'
    plt.savefig(output_filename)
    
    print(f"\n✅ Success! Plot saved to: {output_filename}")
    print("Open this image to see if your VAE is clustering emotions.")

def main():
    global dir 
    dir='val'
    try:
        latents, labels = load_and_align_data(CSV_PATH, LATENTS_PATH)
        plot_tsne(latents, labels)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        dir='val'
    dir='train'
    try:
        latents, labels = load_and_align_data(CSV_PATH, LATENTS_PATH)
        plot_tsne(latents, labels)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()