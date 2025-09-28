# Step 1: Download dataset from Roboflow
from roboflow import Roboflow

# Initialize Roboflow with API key
rf = Roboflow(api_key="je2TnmzCWrwd3g6g83Li")

# Access workspace and project
project = rf.workspace("piero-cic3d").project("detector_de_cascos")

# Select version and download in YOLOv5 format
version = project.version(4)
dataset = version.download("yolov5")
