# 🎧 Deepfake Voice Cloning Detection — Final Year Project (FYP)

### 👥 FYP Group Members
* **Ayesha Arshad** — Group Leader (GL)
* **Faizan Shaikh** — Assistant Group Leader (AGL)
* **Areeba Khan** — Group Member

---

## 📌 Project Overview

This project focuses on the detection of **deepfake audio and AI voice clones** using advanced deep learning. 
Deepfake audio is synthetically generated speech created using generative AI architectures (like Tacotron, Bark, VITS, or ElevenLabs). It poses significant security risks, including impersonation, fraud calls, and misinformation.

Our system takes an uploaded voice recording in **any audio format**, automatically generates a **Mel-spectrogram**, and uses a trained Convolutional Neural Network (CNN) binary classifier to instantly predict whether the voice is:
* 🟢 **REAL** (Natural Human Voice)
* 🚨 **FAKE** (AI-Generated / Cloned Voice)

---

## 🚀 Key Features & Performance Upgrades

We have optimized and enhanced this project to meet professional production and academic standards:

1. **⚡ 100x Faster Performance (Model Caching)**:
   * By caching the pre-trained TensorFlow model in RAM during startup, we eliminated the 5–10 second freeze on every file upload. Predictions are now **instantaneous (milliseconds)**!
2. **📊 Interactive Explainable AI (XAI) Acoustic Dashboard**:
   * Features a live acoustic fingerprint analyzer. When a user clicks **Explain Prediction**, the app dynamically extracts and charts crucial audio features:
     * **Spectral Centroid**: Vocal brightness over time.
     * **Spectral Flatness**: Baseline noise level (to detect vocoder filters).
     * **Zero Crossing Rate**: Signal changes per second (to find synthetic micro-jitter).
3. **🎵 Universal Format Support & Playback**:
   * Accepts voice recordings in **every major format** (WAV, MP3, FLAC, OGG, M4A, AAC, WMA, AIFF) with dynamic MIME type mapping for high-fidelity web player playback.
4. **🛡️ Safe Upload Pipeline**:
   * Robust try-except exception handling protects the backend from crashing if a corrupted or empty file is uploaded, displaying a clean info banner instead.
5. **🧹 Memory Leak Protection**:
   * Fully cleans up Matplotlib figure memory buffers after generating spectrograms to ensure the Streamlit server runs continuously and efficiently.

---

## 📁 Repository Structure

```
Deepfake-Voice-Cloning-Detection/
│
├── Streamlit/
│   ├── app.py                      # Optimized Streamlit Application
│   ├── requirements.txt            # Python Dependencies
│   ├── saved_model/
│   │   └── model/                  # Pre-trained Keras SavedModel (Structure & Weights)
│   └── audio_files/                # Audio uploads and reference samples
│
├── Code/                           # Jupyter Notebooks for model training & evaluation
│   ├── Audio_classifier.ipynb
│   ├── InceptionV3-MobileNet.ipynb
│   ├── Spectrogram-converter.ipynb
│   └── VGG16-Custom CNN-ResNet.ipynb
│
├── img/                            # Project diagrams and UI assets
├── .gitignore                      # Excludes local venv and IDE settings
└── README.md                       # Repository Presentation Page
```

---

## ⚙️ Installation & Setup (Local Run Guide)

Follow these simple steps to set up and run this project locally on your machine.

### 🔹 1. Create a Virtual Environment
Navigate to your project root folder and create a clean virtual environment matching your local system path:
```powershell
python -m venv venv
```

### 🔹 2. Activate the Environment
* **On Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\activate
  ```
* **On macOS/Linux (Terminal)**:
  ```bash
  source venv/bin/activate
  ```

### 🔹 3. Install Dependencies
Install all required packages from the requirements file and upgrade the typing library to ensure interactive charts render flawlessly:
```powershell
pip install -r Streamlit/requirements.txt
pip install --upgrade typing-extensions
```

### 🔹 4. Run the Streamlit Application
Start the server by running:
```powershell
cd Streamlit
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser to play, analyze, and detect audio files!

---

## 🧪 Model & Dataset details

* **Training Target**: Binary classification of voice samples.
* **Feature Processing**: Audio waveforms are transformed into 2D Mel-Spectrogram power matrices, resizing them to `(224, 224)` for input into the neural network.
* **Detections**: Identifies neural vocoder artifacts (high-frequency phase shifts and synthetic grid patterns) that are physically impossible for human vocal tract acoustics to create.
