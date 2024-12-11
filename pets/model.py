if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    results = model.train(data="D:\\GitHub\\ML\\pets\\petsconfig.yaml", epochs=100, imgsz=384)
