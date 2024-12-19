from pathlib import Path
from ultralytics import YOLO
import cv2

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "validationtest" / "best.pt"
video_path = BASE_DIR / "validationtest" / "dog.mp4"
output_path = BASE_DIR / "validationtest" / "dog_annotated.mp4"

model = YOLO(str(model_path))

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
process_fps = 10  
frame_interval = int(fps / process_fps)

out = cv2.VideoWriter(
    str(output_path), 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (frame_width, frame_height)
)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model(frame)
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  

cap.release()
out.release()
cv2.destroyAllWindows()
