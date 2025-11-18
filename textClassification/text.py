from transformers import pipeline
from pathlib import Path
import sys

# --- 1. Emotion Mapping ---
# This map is built for the 'SamLowe' model, which has 28 labels.
# We map its 'neutral' label to our 'calm' category.
TEXT_MODEL_TO_MY_EMOTIONS = {
    # Happy
    'joy': 'happy',
    'amusement': 'happy',
    'excitement': 'happy',
    'love': 'happy',
    'optimism': 'happy',
    'gratitude': 'happy',
    'surprise': 'happy',
    'approval': 'happy',
    
    # Sad
    'sadness': 'sad',
    'disappointment': 'sad',
    'grief': 'sad',
    'disgust': 'sad',
    'remorse': 'sad',

    # Angry
    'anger': 'angry',
    'annoyance': 'angry',
    'fear': 'angry',
    'nervousness': 'angry',
    'disapproval': 'angry',

    # Calm
    'neutral': 'calm',
    'caring': 'calm',
    'relief': 'calm',
    'pride': 'calm',
    'admiration': 'calm',
    'realization': 'calm',
    'curiosity': 'calm',
    'desire': 'calm',
    'confusion': 'calm'
}
DEFAULT_EMOTION = 'calm'

# --- 2. Load Text Emotion Classifier ---
# We are now using the 'SamLowe' model as our primary.
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

print(f"Loading Text Emotion model: '{MODEL_NAME}'...")
print("(This may take a moment and will download ~500MB the first time.)")

try:
    classifier = pipeline(
        "text-classification", 
        model=MODEL_NAME, 
        top_k=1
    )
    print("Text Emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- 3. Start Text Input Loop ---
print("\n--- Text Emotion Tester ---")
print("Type a sentence and press Enter to classify its emotion.")
print("Type 'q' or 'quit' to exit.")

# while True:
#     user_text = input("\nEnter text: ")
    
#     if user_text.lower() in ['q', 'quit']:
#         break
    
#     if not user_text:
#         continue

#     # 1. Analyze text to get the raw emotion
#     try:
#         result = classifier(user_text)
#         raw_emotion = result[0][0]['label']
#         score = result[0][0]['score']
        
#     except Exception as e:
#         print(f"Error analyzing text: {e}. Please try again.")
#         continue
    
#     # 2. Map the raw emotion to one of your 4 categories
#     mapped_emotion = TEXT_MODEL_TO_MY_EMOTIONS.get(raw_emotion, DEFAULT_EMOTION)
    
#     # 3. Print the results
#     print(f"  -> Raw Emotion: '{raw_emotion}' (Confidence: {score*100:.1f}%)")
#     print(f"  -> Mapped Emotion: '{mapped_emotion}'")
        
# # --- 4. Cleanup ---
def predict_emotion(text):
    """Takes a string input and returns the mapped emotion."""
    if not text:
        return "calm"
        
    results = classifier(text)
    raw_emotion = results[0][0]['label']
    mapped_emotion = TEXT_MODEL_TO_MY_EMOTIONS.get(raw_emotion, 'calm')
    return mapped_emotion

print("Exiting.")