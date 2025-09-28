#Pasor 1: Descargar el dataset
from roboflow import Roboflow
rf = Roboflow(api_key="je2TnmzCWrwd3g6g83Li")
project = rf.workspace("piero-cic3d").project("detector_de_cascos")
version = project.version(4)
dataset = version.download("yolov5")
                