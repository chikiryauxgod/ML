from ultralytics import YOLO
import cv2
import sys

def setup_video(video_path, output_path, output_fps):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError("Unable to open video.")
    except IOError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
     
    frame_interval = int(fps / output_fps)

    out = cv2.VideoWriter(
        (output_path), 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        output_fps, 
        (frame_width, frame_height)
    )
    
    return cap, out, frame_interval

def main():
    model_path = ("./ML/validationtest/best.pt")
    video = "rabbit4.mp4"
    video_path = (f"./ML/validationtest/examples/{video}")
    output_path = (f"./ML/validationtest/results/annotated_{video}")
    output_fps = 10 

    model = YOLO(model_path)

    cap, out, frame_interval = setup_video(video_path, output_path, output_fps)

    frame_count = 0
    ret, frame = cap.read()
    while ret:
        if frame_count % frame_interval == 0:
            results = model(frame, conf=0.65)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cv2.imshow("Detection test", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1  
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()