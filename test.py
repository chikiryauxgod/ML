import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(data = 'config.yaml', epochs = 10, imgz = 640)