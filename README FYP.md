# 🎧 Deepfake Voice Cloning Detection – Final Year Project

### **FYP Group Members**
- **Ayesha Arshad (Group Leader)**
- **Faizan Shaikh (Assistant Group Leader)**
- **Areeba Khan**

---

# 📌 Project Overview

This project focuses on detecting **deepfake audio** using deep learning.  
Deepfake audio refers to artificially generated voice recordings produced using advanced AI-based generative models. Such fake audio can be used for impersonation, scam calls, misinformation, and other malicious purposes.

Our system:

1. Takes an uploaded `.wav` file  
2. Converts it into a **mel-spectrogram**  
3. Uses a trained deep learning model to classify audio as:
   - **Real**
   - **Fake**

A fully functional **Streamlit-based web application** is included for live detection.

---

# 🧠 Model Used

A TensorFlow/Keras deep learning model trained on mel-spectrograms of:

- Real speech audio  
- Fake audio generated using voice-cloning algorithms  

The model performs **binary classification**.

---

# 🛠️ Technologies Used

### **Programming Language**
- Python 3.10

### **Libraries**
- Streamlit  
- TensorFlow / Keras  
- Librosa  
- NumPy  
- Matplotlib  
- OpenCV  
- Pandas  

### **Tools**
- Jupyter Notebook  
- Streamlit Web App  
- Virtual Environment  

---

# 🎤 How the System Works

### **1. User Uploads Audio**
A `.wav` file is uploaded through the Streamlit UI.

### **2. Mel-Spectrogram Conversion**
The audio is processed using Librosa to generate a mel-spectrogram.

### **3. Preprocessing**
The spectrogram is resized, normalized, and prepared for model inference.

### **4. Deep Learning Prediction**
The trained model predicts whether the input audio is **real** or **fake**.

### **5. Output Display**
The system shows:
- Audio playback  
- Spectrogram visualization  
- Classification output  

---

# 🖥️ Streamlit Web Application

The UI provides:

✔ Interactive audio player  
✔ Automatic spectrogram visualization  
✔ Instant classification  
✔ Clean interface suitable for demos and deployment  

---

# 📁 Project Structure

```
Deepfake-Audio-Detection/
│
├── Streamlit/
│   ├── app.py
│   ├── requirements.txt
│   ├── saved_model/
│   │   └── model
│   ├── audio_files/
│   └── melspectrogram.png
│
├── Code/
├── img/
├── README.md
└── other system folders (.idea, .venv, etc.)
```

---

# ⚙️ Installation & Setup (Complete Guide)

## 🔹 1. Install Python 3.10  
TensorFlow requires Python 3.10 on Windows.

---

## 🔹 2. Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 🔹 3. Install Dependencies

```powershell
pip install streamlit opencv-python pandas librosa matplotlib Keras-Preprocessing
pip install numpy==1.24.3
pip install protobuf==4.23.4
pip install keras==2.13.1
pip install tensorboard==2.13.0
pip install tensorflow==2.13.0
```

---

## 🔹 4. Run the Streamlit App

```powershell
cd Streamlit
streamlit run app.py
```

Streamlit will open at:

```
http://localhost:8501
```

---

# 🧪 Dataset

The model was trained on:
- Real human speech samples  
- Fake audio generated using AI voice-cloning models  

Mel-spectrograms were used to improve classification accuracy.

---

# 📊 Model Evaluation

The model is evaluated on:
- Accuracy  
- Precision  
- Binary classification performance  

---

# 🎯 Conclusion

This project provides:

- A real-time deepfake audio detection tool  
- A complete pipeline from audio → spectrogram → classification  
- A deployable Streamlit-based interface  
- A strong foundation for future work such as robustness against GAN-based attacks  

---

# 🙌 FYP Group Members

- **Ayesha Arshad (GL)**  
- **Faizan Shaikh (AGL)**  
- **Areeba Khan**
