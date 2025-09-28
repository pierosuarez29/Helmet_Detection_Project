# 🪖 Helmet Detection System – YOLOv5s  

## Overview  
This repository contains a computer vision system designed to **detect safety helmet compliance on construction sites**.  
The project leverages **YOLOv5s** for object detection, enabling both real-time inference via webcam and evaluation on pre-recorded videos.  

The motivation behind this project is to improve occupational safety by monitoring workers’ adherence to helmet regulations.  
By automating detection, the system contributes to accident prevention and supports a stronger safety culture in construction environments.  

---

## 🔑 Key Features  

- 📥 **Dataset Management**: Automated dataset download from Roboflow.  
- 🔢 **Image Counting**: Utility script to verify dataset integrity.  
- 🧠 **Model Training**: Fine-tuned YOLOv5s training pipeline with custom classes (`protected` vs `at risk`).  
- ⚡ **CUDA Support**: Quick test to confirm GPU availability.  
- 🖥️ **Real-Time Detection**: GUI-based interface to process webcam streams or video files.  
- 📊 **Performance**: Achieved **91.9% mAP@0.5**, **0.89 F1-score**, and **110 FPS** in inference speed.  

---

## 📂 Project Structure  

```
├── descargar_dataset.py    # Download dataset from Roboflow
├── contar_imagenes.py      # Count number of images (train/valid/test)
├── entrenar_modelo.py      # Train YOLOv5s with dataset
├── probar_cuda.py          # Check GPU & CUDA availability
├── yolo_utils.py           # Utilities for model loading & inference
├── gui.py                  # Tkinter GUI for webcam/video detection
├── videos_yolo/            # Sample videos for testing
└── README.md               # Project documentation
```

---

## ⚙️ Installation  

```bash
# 1) Clone the repository
git clone https://github.com/your_user/helmet-detection-yolov5.git
cd helmet-detection-yolov5

# 2) Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

# 3) Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage  

1️⃣ **Download dataset**  
```bash
python descargar_dataset.py
```  

2️⃣ **Verify dataset images**  
```bash
python contar_imagenes.py
```  

3️⃣ **Train the model**  
```bash
python entrenar_modelo.py
```  

4️⃣ **Validate CUDA (optional)**  
```bash
python probar_cuda.py
```  

5️⃣ **Run GUI application**  
```bash
python gui.py
```
👉 The GUI allows you to select a **video file** or activate the **webcam** for live detection.  

---

## 📊 Experimental Results  

From the experimental study:  

- **mAP@0.5**: 91.9%  
- **F1-score**: 0.89 at optimal threshold  
- **Inference speed**: 110 FPS on NVIDIA RTX 4060  

These results confirm the model’s suitability for real-world deployment in construction environments.  

---

## ✨ Future Work  

- Extend detection to other PPE such as vests, gloves, and goggles.  
- Integration with CCTV systems and IoT alert platforms.  
- Evaluation under extreme lighting and weather conditions.  
- Migration to YOLOv8 or other state-of-the-art architectures for enhanced accuracy.  

---

## 👨‍💻 Authors  

- **Piero Enrique Suarez Chavez** (Owner, Developer)  
- **Minerva Antonella Paz Bodero** (Co-author, Documentation, Validation)  
