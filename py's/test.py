import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
print(model.predict("C:\\Users\\lewac\\Downloads\\123.jpg"))