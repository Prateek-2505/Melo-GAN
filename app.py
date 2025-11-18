import os
import yaml
import torch
import numpy as np
import io
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS

# --- IMPORT MODULES ---
from textClassification.text import predict_emotion
# We import the generator function and the variable from your camera module
from camera.camera import generate_frames 
import camera.camera as cam_module 

# --- IMPORT GAN ---
from src.gan.models import Generator
from src.gan.feature_encoder import FeatureEncoder
from src.gan.utils import save_piano_roll_to_midi

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

# ==========================================
# 1. LOAD GAN MODELS (This was missing!)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = "config/gan_config.yaml"
CHECKPOINT_PATH = "experiments/gan/checkpoints/gan_final.pth"

print(f"[INIT] Loading GAN models on {DEVICE}...")

# Load Config
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

numeric_dim = cfg.get('NUMERIC_INPUT_DIM', 6)
embed_dim = cfg.get('ENCODER_OUT_DIM', 128)

# Load Architecture
E_num = FeatureEncoder(in_dim=numeric_dim, hidden_dims=cfg.get('ENCODER_HIDDEN'), out_dim=embed_dim, dropout=0.0).to(DEVICE)
G = Generator(noise_dim=cfg['NOISE_DIM'], latent_dim=cfg['LATENT_DIM'], mode=cfg.get('INTEGRATION_MODE'), hidden=cfg.get('GEN_HIDDEN'), max_notes=cfg['MAX_NOTES'], note_dim=cfg['NOTE_DIM'], numeric_embed_dim=embed_dim).to(DEVICE)

# Load Weights
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    G.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
    E_num.load_state_dict(ckpt['E_num'])
    G.eval(); E_num.eval()
    print("[INIT] GAN Models loaded successfully.")
else:
    print("[ERROR] GAN Checkpoint not found! Generation will fail.")

def get_gan_features(emotion):
    """Maps emotion to GAN input features with jitter."""
    emotion = emotion.lower()
    base = None
    if emotion == "happy": base = [1.0, 1.0, 0.8, 0.8, 0.5, 0.5]
    elif emotion == "sad": base = [-1.0, -1.0, -0.5, -0.5, -0.5, -0.5]
    elif emotion == "angry": base = [1.0, -1.0, 1.0, 1.0, -0.8, 0.8]
    elif emotion == "calm": base = [-1.0, 1.0, -0.8, -0.8, 0.5, -0.5]
    
    if base:
        t = torch.tensor([base]).float().to(DEVICE)
        return t + (torch.randn_like(t) * 0.15) # Add jitter
    return torch.randn(1, numeric_dim).to(DEVICE)

# ==========================================
# 2. ROUTES
# ==========================================

@app.route('/')
def index():
    # This serves the index.html we just created
    return render_template('index.html')

@app.route('/get_text_emotion', methods=['POST'])
def get_text_emotion():
    user_text = request.json.get('text')
    emotion = predict_emotion(user_text)
    return jsonify({'emotion': emotion})

@app.route('/video_feed')
def video_feed():
    # Streams the camera frames to the <img> tag
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_camera_emotion')
def get_camera_emotion():
    # Fetches the variable from camera.py
    return jsonify({'emotion': cam_module.current_emotion})

@app.route('/generate', methods=['POST'])
def generate_music():
    emotion = request.json.get('emotion', 'happy')
    print(f"[API] Generating music for: {emotion}")
    
    with torch.no_grad():
        # 1. Prepare inputs
        features = get_gan_features(emotion)
        emb = E_num(features)
        noise = torch.randn(1, cfg['NOISE_DIM']).to(DEVICE)
        latent = torch.zeros(1, cfg['LATENT_DIM']).to(DEVICE)
        
        # 2. Generate
        gen_notes, _ = G(noise, latent, emb)
        notes_cpu = gen_notes.cpu().numpy()[0]
    
    # 3. Save to MIDI
    scale = 'major' if emotion in ['happy', 'calm'] else 'minor'
    bpm = 140 if emotion == 'happy' else 70 if emotion == 'sad' else 160 if emotion == 'angry' else 90
    
    midi_buffer = io.BytesIO()
    save_piano_roll_to_midi(notes_cpu, "temp_gen.mid", bpm=bpm, scale_type=scale)
    
    with open("temp_gen.mid", "rb") as f:
        midi_buffer.write(f.read())
    midi_buffer.seek(0)
    
    return send_file(midi_buffer, mimetype='audio/midi', as_attachment=True, download_name=f'melo_{emotion}.mid')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




























# import os
# import yaml
# import torch
# import numpy as np
# import cv2
# import base64
# from flask import Flask, request, send_file, jsonify,render_template,Response
# from flask_cors import CORS
# from transformers import pipeline
# from tensorflow.keras.models import load_model
# import io
# from textClassification.text import predict_emotion
# from camera.camera import generate_frames, current_emotion
# import camera.camera as cam_module # Access global variable directly

# # Import GAN modules
# from src.gan.models import Generator
# from src.gan.feature_encoder import FeatureEncoder
# from src.gan.utils import save_piano_roll_to_midi



# # We point the template folder to your 'web/templates' directory
# app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
# CORS(app)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[INIT] Running on {DEVICE}")

# # ==========================================
# # 1. LOAD GAN MODELS
# # ==========================================
# CONFIG_PATH = "config/gan_config.yaml"
# CHECKPOINT_PATH = "experiments/gan/checkpoints/gan_final.pth"

# with open(CONFIG_PATH) as f:
#     cfg = yaml.safe_load(f)

# numeric_dim = cfg.get('NUMERIC_INPUT_DIM', 6)
# embed_dim = cfg.get('ENCODER_OUT_DIM', 128)

# E_num = FeatureEncoder(in_dim=numeric_dim, hidden_dims=cfg.get('ENCODER_HIDDEN'), out_dim=embed_dim, dropout=0.0).to(DEVICE)
# G = Generator(noise_dim=cfg['NOISE_DIM'], latent_dim=cfg['LATENT_DIM'], mode=cfg.get('INTEGRATION_MODE'), hidden=cfg.get('GEN_HIDDEN'), max_notes=cfg['MAX_NOTES'], note_dim=cfg['NOTE_DIM'], numeric_embed_dim=embed_dim).to(DEVICE)

# if os.path.exists(CHECKPOINT_PATH):
#     ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
#     G.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
#     E_num.load_state_dict(ckpt['E_num'])
#     G.eval(); E_num.eval()
#     print("[INIT] GAN Models loaded.")
# else:
#     print("[ERROR] GAN Checkpoint not found!")

# # ==========================================
# # 2. LOAD TEXT EMOTION MODEL
# # ==========================================
# TEXT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
# text_classifier = pipeline("text-classification", model=TEXT_MODEL_NAME, top_k=1)
# print("[INIT] Text Emotion Model loaded.")

# TEXT_MAPPING = {
#     'joy': 'happy', 'amusement': 'happy', 'excitement': 'happy', 'love': 'happy',
#     'sadness': 'sad', 'disappointment': 'sad', 'grief': 'sad',
#     'anger': 'angry', 'annoyance': 'angry', 'fear': 'angry',
#     'neutral': 'calm', 'relief': 'calm', 'curiosity': 'calm'
# }

# # ==========================================
# # 3. LOAD CAMERA EMOTION MODELS
# # ==========================================
# FACE_PROTO = "deploy.prototxt.txt"
# FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
# EMOTION_MODEL_H5 = "mini_xception.h5"

# face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# emotion_model_cv = load_model(EMOTION_MODEL_H5, compile=False)
# EMOTION_INPUT_SIZE = emotion_model_cv.input_shape[1:3] # (48, 48) usually
# print("[INIT] Camera Models loaded.")

# CV_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# CV_MAPPING = {
#     'happy': 'happy', 'sad': 'sad', 'angry': 'angry', 'fear': 'angry',
#     'neutral': 'calm', 'surprise': 'happy', 'disgust': 'sad'
# }

# # ==========================================
# # ROUTES
# # ==========================================

# @app.route('/predict/text', methods=['POST'])
# def predict_text():
#     data = request.json
#     text = data.get('text', '')
#     if not text: return jsonify({'error': 'No text provided'}), 400
    
#     results = text_classifier(text)
#     raw_label = results[0][0]['label']
#     mapped_emotion = TEXT_MAPPING.get(raw_label, 'calm')
    
#     return jsonify({'raw': raw_label, 'emotion': mapped_emotion})

# @app.route('/predict/image', methods=['POST'])
# def predict_image():
#     data = request.json
#     image_data = data.get('image', '') # Base64 string
    
#     try:
#         # Decode image
#         encoded_data = image_data.split(',')[1]
#         nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#         face_net.setInput(blob)
#         detections = face_net.forward()

#         # Find best face
#         best_conf = 0.0
#         best_box = None
        
#         for i in range(0, detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#             if confidence > 0.5 and confidence > best_conf:
#                 best_conf = confidence
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 best_box = box.astype("int")
        
#         if best_box is None:
#             return jsonify({'emotion': 'calm', 'message': 'No face detected, defaulting to calm.'})

#         (startX, startY, endX, endY) = best_box
#         # Ensure crop is within bounds
#         startX, startY = max(0, startX), max(0, startY)
#         endX, endY = min(w, endX), min(h, endY)
        
#         face_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[startY:endY, startX:endX]
#         resized = cv2.resize(face_roi, EMOTION_INPUT_SIZE)
#         normalized = resized.astype("float") / 255.0
#         expanded = np.expand_dims(normalized, axis=0)
#         final_in = np.expand_dims(expanded, axis=-1)

#         preds = emotion_model_cv.predict(final_in, verbose=0)[0]
#         label = CV_LABELS[preds.argmax()]
#         mapped = CV_MAPPING.get(label, 'calm')
        
#         return jsonify({'raw': label, 'emotion': mapped})

#     except Exception as e:
#         print(f"Image Error: {e}")
#         return jsonify({'error': str(e)}), 500

# # Helper to map emotion to GAN features (From previous chat)
# def get_features(emotion):
#     emotion = emotion.lower()
#     if emotion == "happy": return torch.tensor([[1.0, 1.0, 0.8, 0.8, 0.5, 0.5]]).float().to(DEVICE)
#     elif emotion == "sad": return torch.tensor([[-1.0, -1.0, -0.5, -0.5, -0.5, -0.5]]).float().to(DEVICE)
#     elif emotion == "angry": return torch.tensor([[1.0, -1.0, 1.0, 1.0, -0.8, 0.8]]).float().to(DEVICE)
#     elif emotion == "calm": return torch.tensor([[-1.0, 1.0, -0.8, -0.8, 0.5, -0.5]]).float().to(DEVICE)
#     else: return torch.randn(1, numeric_dim).to(DEVICE)

# @app.route('/generate', methods=['POST'])
# def generate_music():
#     emotion = request.json.get('emotion', 'happy')
#     print(f"[API] Generating for: {emotion}")
    
#     with torch.no_grad():
#         # Jitter logic included
#         base_feats = get_features(emotion)
#         jitter = torch.randn_like(base_feats) * 0.15
#         emb = E_num(base_feats + jitter)
#         noise = torch.randn(1, cfg['NOISE_DIM']).to(DEVICE)
#         latent = torch.zeros(1, cfg['LATENT_DIM']).to(DEVICE)
#         gen_notes, _ = G(noise, latent, emb)
    
#     midi_buffer = io.BytesIO()
    
#     # Scale logic
#     scale = 'major' if emotion in ['happy', 'calm'] else 'minor'
#     bpm = 140 if emotion == 'happy' else 70 if emotion == 'sad' else 160 if emotion == 'angry' else 90
    
#     save_piano_roll_to_midi(gen_notes.cpu().numpy()[0], "temp.mid", bpm=bpm, scale_type=scale)
    
#     with open("temp.mid", "rb") as f: midi_buffer.write(f.read())
#     midi_buffer.seek(0)
    
#     return send_file(midi_buffer, mimetype='audio/midi', as_attachment=True, download_name=f'{emotion}.mid')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)