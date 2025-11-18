import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import os

# --- 1. Configuration ---

# --- Paths to your new models ---
# FACE_PROTO = "deploy.prototxt.txt"
# FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
# EMOTION_MODEL_PATH =  "mini_xception.h5"

# 1. Fix Paths (Must be absolute to work from root folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Path to /camera folder
FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt.txt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "mini_xception.h5") 

# --- 2. Emotion Mapping ---
# The Mini-Xception model (trained on FER2013) outputs 7 emotions
MINI_XCEPTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# We map them to your 4 categories
XCEPTION_TO_MY_EMOTIONS = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'neutral': 'calm',
    'fear': 'angry',
    'surprise': 'happy',
    'disgust': 'sad'
}

# --- 3. Load Your Models ---
print("Loading OpenCV DNN Face Detector...")
try:
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    print("Face Detector loaded.")
except cv2.error as e:
    print(f"--- FATAL ERROR ---")
    print(f"Could not load face detector models. Make sure you have:")
    print(f"1. {FACE_PROTO}")
    print(f"2. {FACE_MODEL}")
    print(f"In your root folder. Error: {e}")
    exit()

print("Loading Mini-Xception Keras model...")
try:
    emotion_model = load_model(EMOTION_MODEL_PATH,compile = False)
    # Get model's expected input size (e.g., (48, 48) or (64, 64))
    EMOTION_INPUT_SIZE = emotion_model.input_shape[1:3]
    print(f"Emotion model loaded. Expects input shape: {EMOTION_INPUT_SIZE}")
except Exception as e:
    print(f"--- FATAL ERROR ---")
    print(f"Could not load emotion model from: {EMOTION_MODEL_PATH}")
    print(f"Error: {e}")
    exit()

# --- 4. Start Real-Time Loop ---
# print("\nStarting webcam... Press 'q' to quit.")
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# last_mapped_emotion = 'calm'
current_emotion = "calm"

        
def generate_frames():
    global current_emotion
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        best_conf = 0.0
        best_box = None

        # Face Detection Logic
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_conf:
                best_conf = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_box = box.astype("int")
        
        if best_box is not None:
            (startX, startY, endX, endY) = best_box
            # Ensure box is within frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            try:
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size > 0:
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, EMOTION_INPUT_SIZE)
                    normalized = resized.astype("float") / 255.0
                    expanded = np.expand_dims(normalized, axis=0)
                    final_in = np.expand_dims(expanded, axis=-1)

                    preds = emotion_model.predict(final_in, verbose=0)[0]
                    label = MINI_XCEPTION_LABELS[preds.argmax()]
                    current_emotion = XCEPTION_TO_MY_EMOTIONS.get(label, 'calm')

                    # Draw box and label
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, current_emotion, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception:
                pass

        # Encode frame for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
        
        
        
        
        

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     (h, w) = frame.shape[:2]
    
#     # 1. Create a blob from the frame
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
#     # 2. Pass blob through the face detector
#     face_net.setInput(blob)
#     detections = face_net.forward()

#     # Find the face with the highest confidence
#     best_confidence = 0.0
#     best_box = None
    
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
        
#         if confidence > 0.5 and confidence > best_confidence:
#             best_confidence = confidence
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             best_box = box.astype("int")

#     # If a face was found...
#     if best_box is not None:
#         (startX, startY, endX, endY) = best_box
        
#         try:
#             # 1. Extract the face ROI (Region of Interest)
#             face_roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[startY:endY, startX:endX]
            
#             # 2. Preprocess for Mini-Xception
#             face_roi_resized = cv2.resize(face_roi_gray, EMOTION_INPUT_SIZE)
#             face_roi_normalized = face_roi_resized.astype("float") / 255.0
#             face_roi_expanded = np.expand_dims(face_roi_normalized, axis=0)
#             face_roi_final = np.expand_dims(face_roi_expanded, axis=-1) # (1, 48, 48, 1)

#             # 3. Predict emotion
#             preds = emotion_model.predict(face_roi_final, verbose=0)[0]
#             emotion_idx = preds.argmax()
#             raw_emotion = MINI_XCEPTION_LABELS[emotion_idx]
            
#             # 4. Map to your 4 emotions
#             last_mapped_emotion = XCEPTION_TO_MY_EMOTIONS.get(raw_emotion, 'calm')

#         except Exception as e:
#             # Error during preprocessing (e.g., face too close to edge)
#             # We just use the last known emotion
#             pass 
        
#         # Draw the mapped emotion on the frame
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         cv2.putText(frame, last_mapped_emotion, (startX, startY - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Real-Time Emotion (DNN + Mini-Xception) - Press "q" to quit', frame)

#     # --- 5. Check for Key Press ---
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
        
        
# --- 6. Cleanup ---

# cap.release()
# cv2.destroyAllWindows()
# print("Webcam closed. Exiting.")