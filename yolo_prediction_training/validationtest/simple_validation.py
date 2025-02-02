from ultralytics import YOLO

def main():

    image = "rabbit.jpg"
    video = "dog.mp4"
    
    model = YOLO("./ML/validationtest/best.pt")
    input_path = (f"./ML/validationtest/examples/{image}")
    output_path = ("./ML/validationtest/results")

    try:
        result = model.predict(input_path, save=True, imgsz=640, conf=0.65, project=output_path)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()