import argparse
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


def xy_from_box(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def main(args):
    # Load YOLO model
    model = YOLO(args.weights)

    # Resolve tracker path
    tracker_path = args.tracker
    if tracker_path in ["strongsort.yaml", "botsort.yaml"]:
        tracker_path = f"ultralytics/cfg/trackers/{tracker_path}"

    generator = model.track(
        source=args.source,
        tracker=tracker_path,
        device=args.device,
        conf=args.min_conf,
        persist=True,
        stream=True
    )

    # Setup video writer
    out = None
    fps = 30
    cap = cv2.VideoCapture(args.source)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                "tracked_output.mp4",
                fourcc,
                fps,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
    cap.release()

    last_ball_pos = None
    frame_idx = 0

    # Trajectory storage (x, y, age)
    trajectory = deque()
    max_age_frames = int(args.traj_seconds * fps)

    # For prediction
    recent_positions = deque(maxlen=5)  # store recent detections
    miss_frames = 0
    max_prediction_frames = int(args.max_predict_seconds * fps)

    for results in generator:
        frame = results.orig_img.copy()
        detected_this_frame = False

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            xyxy = boxes.xyxy.cpu().numpy()

            # Filter only ball detections
            ball_indices = [i for i in range(len(xyxy)) if clss[i] == args.ball_class_id]
            if ball_indices:
                # Keep highest confidence
                best_idx = max(ball_indices, key=lambda i: confs[i])

                box = xyxy[best_idx]
                cx, cy = xy_from_box(box)
                conf = float(confs[best_idx])

                # ---- Anti-teleport ----
                if last_ball_pos is not None:
                    dist = np.hypot(cx - last_ball_pos[0], cy - last_ball_pos[1])
                    if dist > args.max_jump:
                        # Skip update but keep prediction running
                        pass
                    else:
                        last_ball_pos = (cx, cy)
                        detected_this_frame = True
                else:
                    last_ball_pos = (cx, cy)
                    detected_this_frame = True

                if detected_this_frame:
                    recent_positions.append((cx, cy))
                    trajectory.append((cx, cy, 0))
                    miss_frames = 0

        # ---- Prediction if missed ----
        if not detected_this_frame and last_ball_pos is not None:
            miss_frames += 1
            if 1 < len(recent_positions) <= 5 and miss_frames <= max_prediction_frames:
                # Linear extrapolation from last two positions
                (x1, y1), (x2, y2) = recent_positions[-2], recent_positions[-1]
                dx, dy = x2 - x1, y2 - y1
                pred_x = int(last_ball_pos[0] + dx)
                pred_y = int(last_ball_pos[1] + dy)
                last_ball_pos = (pred_x, pred_y)
                trajectory.append((pred_x, pred_y, 0))
            elif miss_frames > max_prediction_frames:
                last_ball_pos = None  # stop predicting

        # ---- Draw ball ----
        if last_ball_pos is not None:
            cv2.circle(frame, last_ball_pos, 12, (0, 180, 255), 2)

        # ---- Draw fading trajectory ----
        new_traj = deque()
        for (x, y, age) in trajectory:
            age += 1
            if age <= max_age_frames:
                alpha = max(0, 1 - (age / max_age_frames))
                color = (
                    int(0 * (1 - alpha) + 0 * alpha),
                    int(180 * alpha),
                    int(255 * alpha)
                )
                cv2.circle(frame, (x, y), 4, color, -1)
                new_traj.append((x, y, age))
        trajectory = new_traj

        # Output
        if args.save:
            out.write(frame)
        if args.view:
            cv2.imshow("Ball Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if args.save:
        out.release()
    if args.view:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="video source")
    parser.add_argument("--weights", type=str, required=True, help="YOLO model weights")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    parser.add_argument("--ball-class-id", type=int, default=0, help="Class ID of the ball")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--view", action="store_true", help="Display video while processing")
    parser.add_argument("--max-jump", type=float, default=300, help="Max allowed pixel jump per frame")
    parser.add_argument("--traj-seconds", type=float, default=1.5, help="How long to keep trajectory in seconds")
    parser.add_argument("--max-predict-seconds", type=float, default=0, help="Max seconds to keep predicting when ball is lost")
    args = parser.parse_args()

    main(args)
