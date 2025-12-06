# 🎶 Real‑Time Emotion‑Aware Music Generator (Full README)

Welcome to the emotional command center of your AI music universe. This repository fuses **computer vision**, **NLP emotion detection**, and a **GAN-powered symbolic music generator** into one seamless pipeline that reacts to your expressions *and* your text — crafting original MIDI melodies in real time.

Camera → Text → Emotion → GAN → MIDI → Magic.

This README gives you:
- A full breakdown of every module in the repo
- Architecture diagrams (ASCII-style)
- How the GAN, Emotion Discriminator, and Dataset Pipeline fit together
- Installation & running guide
- A summary of requirements based on your `requirements.txt` fileciteturn0file4

---

# 🚀 1. Project Overview
This system is a **multimodal emotional soundtrack engine**. It listens to your face, reads your text, and generates MIDI that reflects those emotions.

It brings together three major subsystems:
1. **Real‑Time Camera Emotion Detection** — Facial expressions → emotion
2. **Text‑Based Emotion Classification** — Typed messages → emotion
3. **MELO‑GAN (WGAN‑GP)** — Emotion‑conditioned symbolic music generation

Each of them is independently testable but fully integrated via `app.py`.

---

# 🧩 2. Repository Structure
Below is a clean summary of the key modules.

## 🎥 2.1 camera.py — Real‑Time Facial Emotion Detection
Handles live webcam emotion detection using:
- OpenCV SSD Face Detector
- Mini‑Xception CNN trained on FER‑2013

Emotion mapping (7 → 4 classes):
```
angry, fear → angry
sad, disgust → sad
happy, surprise → happy
neutral → calm
```

Outputs:
- Global variable: `current_emotion`
- Video stream via `generate_frames()`

Source reference: camera.py details (from provided docs) fileciteturn0file0

---

## 📝 2.2 text.py — Text‑Based Emotion Classification
Uses the **SamLowe/roberta-base-go_emotions** model to classify 28 fine-grained emotions → 4 macro categories.

Exposes:
```python
def predict_emotion(text):
    return mapped_emotion
```

This powers the `/get_text_emotion` endpoint.

Source reference: text.py section in your docs fileciteturn0file0

---

## 🧠 2.3 MELO‑GAN — Emotion‑Driven Music Generator
This subsystem includes:
- WGAN‑GP Generator & Critic
- Feature Encoder for conditioning vectors
- Emotion Discriminator (ED) for emotion supervision

Full architecture & file-level explanations come from your MELO‑GAN docs fileciteturn0file1.

### Components
- `feature_encoder.py` — Converts numeric conditioning → latent embedding
- `models.py` — GAN generator & critic
- `train_gan.py` — Full training pipeline
- `test_gan.py` — Generates samples based on chosen emotion

Outputs:
- Pitch, velocity, duration, step → MIDI
- Emotion‑conditioned melodies using scales, BPM, and instrument rules

---

## 🎛 2.4 Emotion Discriminator (ED)
Used during GAN training for emotional correctness.

Implements:
- Conv1D notes encoder
- MLP classifier
- Supports latent and notes input

Your ED documentation confirms:
- Proper shape handling
- Correct training loop
- Correct checkpoint system

Source: ED README fileciteturn0file2

---

## 🎼 2.5 Dataset Pipeline — EMOPIA + VGMIDI Unified Dataset
Your dataset pipeline is fully automated and model-ready.

Stages:
1. Label merging → unified 4‑emotion taxonomy
2. Manifest building
3. MIDI preprocessing → (512 × 4 note matrices)
4. Train/val/test splitting with feature scaling

Source: Dataset Pipeline README fileciteturn0file3

---

# 🧬 3. System Architecture
```
                +---------------------+
 Webcam ------> |   camera.py         |
                | SSD + MiniXception  |
                | current_emotion     |
                +----------+----------+
                           |
                           v
                +---------------------+
 User Text ---> |    text.py          |
                | Transformer Emotion |
                +----------+----------+
                           |
                           v
                +------------------------------+
                |            app.py            |
                |  - Flask Backend             |
                |  - GAN Conditioning          |
                |  - Music Theory Engine       |
                |  - MIDI Generation           |
                +------------------------------+
                           |
                           v
                   Generated MIDI 🎵
```

---

# 🎼 4. Emotion → Music Mapping
| Emotion | Scales | BPM Range | Instruments |
|---------|--------|------------|-------------|
| **happy** | major, lydian, pentatonic | 120–150 | bright piano, celesta |
| **sad** | minor, dorian | 60–85 | harp, nylon guitar |
| **angry** | blues, minor | 140–170 | electric guitar, accordion |
| **calm** | major, lydian | 70–95 | vibraphone, music box |

These rules ensure that even without AI magic, melodies have the correct emotional feel.

---

# 🧪 5. API Endpoints (Flask)
### **GET /**
Loads the UI.

### **GET /video_feed**
Live MJPEG webcam stream.

### **GET /get_camera_emotion**
Returns the latest facial emotion.

### **POST /get_text_emotion**
Returns text‑based emotion.

### **POST /generate**
Runs the entire pipeline → returns a MIDI file.

---

# ⚙️ 6. Installation & Setup
### 1. Install Dependencies
Your `requirements.txt` includes everything required for:
- Deep learning: PyTorch, TensorFlow
- CV: OpenCV, RetinaFace, MTCNN
- NLP: Transformers, tokenizers
- Music: pretty_midi, mido
- Flask backend

Full file referenced here: requirements.txt fileciteturn0file4

Install:
```bash
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python app.py
```

### 3. Open the App
Go to:
```
http://localhost:5000
```

You now have:
- Real-time emotion detection
- Text emotion classification
- GAN-driven MIDI generation

---

# 🧰 7. How to Train the GAN (Optional)
```bash
python train_gan.py --config config/gan_config.yaml --ed_config config/ed_config.yaml
```

Generate samples:
```bash
python test_gan.py --emotion happy --samples 3 --out outputs/
```

---

# 📦 8. Dataset Usage
Load a processed .npz file:
```python
import numpy as np
x = np.load("dataset/processed/example.npz")
notes = x["notes"]
numeric = x["numeric_features"]
emotion = x["emotion"]
```

---

# 🧠 9. Why This System Is Special
This repo brings together:
- Real‑time multimodal emotional AI
- Transformer NLP
- GAN-based symbolic music generation
- A fully normalized, unified emotion‑labeled dataset
- Flask backend for instant deployment

It’s modular, scalable, and production-ready.

---

# 🏁 10. Final Notes
If you want:
- A polished GitHub front-page badge layout
- A CI/CD pipeline
- Docker deployment
- Auto-download models script
- Architecture diagrams (PNG/SVG)

Tell me — I’ll handle it.

Let’s make this AI composer legendary. 🎵🔥

