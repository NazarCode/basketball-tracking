from ultralytics import YOLO
import cv2
from google.colab import drive # For Google Colab environment
from IPython.display import Video # For displaying video inline in Colab

# --- Mount Google Drive (run once) ---
drive.mount('/content/drive') # Mounts your Google Drive to access files. Only needed if using Google Colab.

# --- Load your trained model ---
model = YOLO("/content/drive/MyDrive/basketball_tracking/yolo_checkpoints/yolov8l_ball_basket_person/weights/best.pt") # Replace with your model path

# --- Open video file ---
video_path = "/content/drive/MyDrive/basketball_tracking/test.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path) 

# --- Get video properties ---
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Video writer for saving output ---
output_path = "/content/test_output.mp4"
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run YOLOv8 inference ---
    results = model(frame, conf=0.25, iou=0.45, verbose=False)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

cap.release()
out.release()

print("✅ Done! Saved as", output_path)

# --- Display the output video inline ---
Video(output_path, embed=True)
