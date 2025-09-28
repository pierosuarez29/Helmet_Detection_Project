import os, sys, subprocess
from pathlib import Path

# Define base directories
BASE_DIR = Path(__file__).resolve().parent
Y5_DIR = BASE_DIR / "yolov5"
DATA_YAML = BASE_DIR / "Detector_de_Cascos-4" / "data.yaml"

# 1) Clone YOLOv5 repository if it does not already exist.
#    This ensures the YOLOv5 training scripts are available locally.
if not Y5_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(Y5_DIR)], check=True)

# 2) Install missing dependencies if required.
#    The YOLOv5 repository provides a requirements.txt file
#    that lists all necessary Python packages.
try:
    import seaborn  # This library is often missing by default.
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(Y5_DIR / "requirements.txt")])

# 2.5) Important note regarding data.yaml configuration
# ------------------------------------------------------
# To ensure YOLOv5 can properly locate the dataset, the paths in
# "Detector_de_Cascos-4/data.yaml" must be defined relative to the
# "yolov5" directory, since train.py is executed from inside that folder.
#
# Example of incorrect configuration (will fail):
#   train: Detector_de_Cascos-4/train/images
#   val:   Detector_de_Cascos-4/valid/images
#   test:  Detector_de_Cascos-4/test/images
#
# Correct configuration:
#   train: ../Detector_de_Cascos-4/train/images
#   val:   ../Detector_de_Cascos-4/valid/images
#   test:  ../Detector_de_Cascos-4/test/images
#
# Using relative paths in this way guarantees portability across
# different machines and environments.
# ------------------------------------------------------

# 3) Launch YOLOv5 training.
#    The subprocess is executed with cwd set to the yolov5 directory
#    to ensure that paths are resolved correctly.
subprocess.run(
    [
        sys.executable, "train.py",
        "--img", "640",                  # Input image size
        "--batch", "16",                 # Batch size
        "--epochs", "50",                # Number of training epochs
        "--data", str(DATA_YAML.resolve()),  # Path to dataset configuration
        "--weights", "yolov5s.pt",       # Pretrained weights (YOLOv5s)
        "--name", "casco_detector_v4"    # Experiment name (for saving results)
    ],
    cwd=str(Y5_DIR),  # Run inside the yolov5/ directory
    check=True
)
