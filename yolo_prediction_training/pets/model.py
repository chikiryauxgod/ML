if __name__ == "__main__":
    from ultralytics import YOLO
    BASE_DIR = Path(__file__).resolve().parent

    yaml_path = BASE_DIR / "pets" / "petsconfig.yaml"
    model = YOLO("yolov8n.pt")
    results = model.train(data= yaml_path, epochs=100, imgsz=384)
