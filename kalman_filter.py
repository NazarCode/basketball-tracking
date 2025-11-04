from ultralytics import YOLO
import cv2
import numpy as np

# Load models
ball_model = YOLO("ball.pt")
court_model = YOLO("court.pt")

# Parameters
VIDEO_PATH = "test.mp4"
COURT_DETECT_INTERVAL = 10   # detect court every N frames
CONF_THRESH = 0.2

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
court_polygon = None

# BoT-SORT tracker (use YOLOv8 track API)
tracker_config = "botsort.yaml"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect court every N frames
    if frame_idx % COURT_DETECT_INTERVAL == 0:
        court_results = court_model(frame, conf=CONF_THRESH)
        if court_results and len(court_results) > 0:
            # Assuming court class is 3 and bbox format is xyxy
            for r in court_results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) == 2:  # court class
                        x1, y1, x2, y2 = box.cpu().numpy()
                        court_polygon = np.array([
                            [int(x1), int(y1)],
                            [int(x2), int(y1)],
                            [int(x2), int(y2)],
                            [int(x1), int(y2)]
                        ])
                        break

    # Track balls
    if court_polygon is not None:
        # Run ball detection
        ball_results = ball_model(frame, conf=CONF_THRESH)
        ball_boxes = []
        for r in ball_results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Check if the center of the ball is inside the court
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                if cv2.pointPolygonTest(court_polygon, (cx, cy), False) >= 0:
                    ball_boxes.append([x1, y1, x2, y2])

        # Use YOLOv8 tracker for filtered balls
        if len(ball_boxes) > 0:
            # Create temporary frame with only valid balls for tracker
            # YOLOv8 track API currently works directly on video, so you may need to
            # implement your own BoT-SORT update step or filter post-tracking
            pass  # placeholder for integrating filtered boxes with BoT-SORT

        # Draw court
        cv2.polylines(frame, [court_polygon], isClosed=True, color=(0,255,0), thickness=2)

        # Draw balls
        for box in ball_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    cv2.imshow("Tracked Ball in Court", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
