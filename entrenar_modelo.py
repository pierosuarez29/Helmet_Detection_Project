import os, sys, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
Y5_DIR = BASE_DIR / "yolov5"
DATA_YAML = BASE_DIR / "Detector_de_Cascos-4" / "data.yaml"

# 1) Clona YOLOv5 si falta
if not Y5_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(Y5_DIR)], check=True)

# 2) Instala requirements si falta algo típico
try:
    import seaborn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(Y5_DIR / "requirements.txt")])

# 2.5) 🔧 Nota sobre data.yaml
# ---------------------------------------
# Para que YOLOv5 encuentre tu dataset correctamente, 
# asegúrate de que las rutas en "Detector_de_Cascos-4/data.yaml" 
# usen un nivel hacia arriba (../), ya que train.py se ejecuta 
# dentro de la carpeta "yolov5".
#
# ❌ Ejemplo que falla:
#   train: Detector_de_Cascos-4/train/images
#   val:   Detector_de_Cascos-4/valid/images
#   test:  ../test/images
#
# ✅ Ejemplo correcto:
#   train: ../Detector_de_Cascos-4/train/images
#   val:   ../Detector_de_Cascos-4/valid/images
#   test:  ../Detector_de_Cascos-4/test/images
#
# De esta manera, las rutas siempre funcionarán en cualquier PC.
# ---------------------------------------

# 3) Lanza el entrenamiento (cambiando cwd para que yolov5 resuelva bien las rutas)
subprocess.run(
    [
        sys.executable, "train.py",
        "--img", "640",
        "--batch", "16",
        "--epochs", "50",
        "--data", str(DATA_YAML.resolve()),
        "--weights", "yolov5s.pt",
        "--name", "casco_detector_v4"
    ],
    cwd=str(Y5_DIR),  # ejecuta dentro de yolov5/
    check=True
)
