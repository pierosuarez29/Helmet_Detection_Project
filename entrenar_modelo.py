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

# 2.5) Note on data.yaml configuration
# ---------------------------------------
# To ensure YOLOv5 correctly locates the dataset, 
# verify that the paths specified in "Detector_de_Cascos-4/data.yaml"
# are defined relative to the execution of train.py, which runs
# inside the "yolov5" directory.
#
# Incorrect example (will fail):
#   train: Detector_de_Cascos-4/train/images
#   val:   Detector_de_Cascos-4/valid/images
#   test:  Detector_de_Cascos-4/test/images
#
# Correct example:
#   train: ../Detector_de_Cascos-4/train/images
#   val:   ../Detector_de_Cascos-4/valid/images
#   test:  ../Detector_de_Cascos-4/test/images
#
# Using relative paths in this manner ensures portability 
# across different systems and environments.
# ---------------------------------------
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
