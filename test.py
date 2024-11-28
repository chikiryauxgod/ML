import os
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "C:\\Users\\lewac\\Downloads\\S600xU_2x.webp"

# Run inference on the source
results = model(source)  # list of Results objects