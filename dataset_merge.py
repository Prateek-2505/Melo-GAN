import pandas as pd

def process_emopia_labels(filepath="label.csv"):
    """
    Loads the EMOPIA label file, standardizes emotion labels,
    and selects the correct columns.
    """
    try:
        df = pd.read_csv(filepath)

        # Q1 (High V, High A) = Happy
        # Q2 (Low V, High A) = Angry
        # Q3 (Low V, Low A) = Sad
        # Q4 (High V, Low A) = Calm
        emotion_map = {
            1: 'happy',
            2: 'angry',
            3: 'sad',
            4: 'calm'
        }

        df['emotion'] = df['4Q'].map(emotion_map)

        # Select and rename the essential columns
        df_clean = df[['ID', 'emotion']].copy()
        df_clean.rename(columns={'ID': 'file_key'}, inplace=True)

        return df_clean

    except FileNotFoundError:
        print(f"Error: EMOPIA label file not found at {filepath}")
        return None

def process_vgmidi_labels(filepath="vgmidi_labelled.csv"):
    """
    Loads the VGMIDI label file, standardizes emotion labels,
    and selects the correct columns.
    """
    try:
        df = pd.read_csv(filepath)

        # Map Valence/Arousal values (1 or -1) to emotions
        def map_vgmidi_emotion(valence, arousal):
            if valence == 1 and arousal == 1:
                return 'happy'
            elif valence == -1 and arousal == 1:
                return 'angry'
            elif valence == -1 and arousal == -1:
                return 'sad'
            elif valence == 1 and arousal == -1:
                return 'calm'
            return 'unknown'

        df['emotion'] = df.apply(
            lambda row: map_vgmidi_emotion(row['valence'], row['arousal']),
            axis=1
        )

        # Select and rename the essential columns
        df_clean = df[['midi', 'emotion']].copy()
        df_clean.rename(columns={'midi': 'file_key'}, inplace=True)

        return df_clean

    except FileNotFoundError:
        print(f"Error: VGMIDI label file not found at {filepath}")
        return None

# --- Main Execution ---

# 1. Process both label files
df_emopia = process_emopia_labels("label.csv")
df_vgmidi = process_vgmidi_labels("vgmidi_labelled.csv")

if df_emopia is not None and df_vgmidi is not None:

    print("--- Clean EMOPIA Labels (Head) ---")
    print(df_emopia.head())
    print("\n" + "="*40 + "\n")

    print("--- Clean VGMIDI Labels (Head) ---")
    print(df_vgmidi.head())
    print("\n" + "="*40 + "\n")

    # 2. Combine into one master DataFrame
    df_combined = pd.concat([df_emopia, df_vgmidi], ignore_index=True)

    print("--- Combined DataFrame (Head) ---")
    print(df_combined.head())
    print("\n--- Combined DataFrame (Tail) ---")
    print(df_combined.tail())

    print("\n--- Combined DataFrame Info ---")
    df_combined.info()

    # 3. Save the combined DataFrame to a new CSV file
    output_filename = "combined_standardized_labels.csv"
    df_combined.to_csv(output_filename, index=False)

    print(f"\nSuccessfully combined and saved labels to: {output_filename}")
    print(f"This file contains {len(df_combined)} total samples.")