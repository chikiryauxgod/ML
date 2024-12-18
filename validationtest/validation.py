from ultralytics import YOLO
import cv2

model = YOLO("D:\\GitHub\\ML\\validationtest\\best.pt")  
video_path = "D:\\GitHub\\ML\\validationtest\\dog.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error.")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
process_fps = 10  
frame_interval = int(fps / process_fps) 


output_path = "D:\\GitHub\\ML\\validationtest\\dog_annotated.mp4"
out = cv2.VideoWriter(
    output_path, 
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
