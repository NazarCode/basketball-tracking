"""
Full Improved Basketball Team Color Classifier
=============================================

This script:
 - Uses two YOLO models: players_basket.pt (players class=1) and ball.pt (ball class=0)
 - Samples blurred, saturated, non-skin pixels from player torso ROIs
 - Clusters colors in LAB space during a learning phase (samples near the ball)
 - Converts cluster centers to HSV for distance checks and visualization
 - Performs per-player temporal smoothing and simple tracking by proximity
 - Excludes referees and near-black uniforms
 - Continues small, conservative centroid refinement after lock-in

Requirements:
    pip install ultralytics opencv-python numpy scikit-learn

Usage:
    - Edit the MODEL paths and VIDEO_SOURCE at the top if necessary
    - Press 'r' to reset learning; 'q' to quit

Note: this is a best-effort single-file prototype. You may want to integrate
this logic into a more robust multi-object tracker (DeepSORT / ByteTrack) for
production-grade identity stability.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque, defaultdict
import time

# ----------------------
# Config
# ----------------------
PLAYER_MODEL_PATH = 'players_basket.pt'
BALL_MODEL_PATH = 'ball.pt'
VIDEO_SOURCE = 'test4.mp4'  # 0 for webcam or path to video file

CONF_PLAYER = 0.4
CONF_BALL = 0.25
IOU_THRESH = 0.45

LEARNING_FRAMES = 25
MIN_LEARN_SAMPLES = 20
SAMPLES_PER_FRAME_MAX = 6
PROXIMITY_RADIUS_FRAC = 0.12

MIN_SAT = 45
MIN_VAL = 50

TORSO_TOP_FRAC = 0.28
TORSO_BOTTOM_FRAC = 0.72
TORSO_WIDTH_FRAC = 0.65

SMOOTH_HISTORY = 6
BLACK_VAL_THRESH = 50
BLACK_SAT_THRESH = 60

CENTROID_UPDATE_MOMENTUM = 0.02  # small momentum to slowly adapt centroids

# Visualization
DRAW_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------
# Utilities
# ----------------------

def bbox_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return np.array([int((x1 + x2) / 2), int((y1 + y2) / 2)])


def bbox_area(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_torso_roi(frame, xyxy):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 8 or bh <= 16:
        return None
    torso_top = y1 + int(bh * TORSO_TOP_FRAC)
    torso_bottom = y1 + int(bh * TORSO_BOTTOM_FRAC)
    torso_width = int(bw * TORSO_WIDTH_FRAC)
    torso_x1 = x1 + (bw - torso_width) // 2
    torso_x2 = torso_x1 + torso_width
    torso_x1 = max(0, torso_x1)
    torso_y1 = max(0, torso_top)
    torso_x2 = min(w - 1, torso_x2)
    torso_y2 = min(h - 1, torso_bottom)
    if torso_x2 <= torso_x1 or torso_y2 <= torso_y1:
        return None
    roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
    # reject extremely small
    if roi.shape[0] < 6 or roi.shape[1] < 6:
        return None
    return roi


def dominant_color_lab(roi_bgr, min_pixels=60, k=2):
    """Return (dom_hsv, dom_lab) or (None, None).
    dom_hsv: uint8 HSV triple
    dom_lab: float32 LAB triple (used for clustering)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, None
    # Downsize and blur
    roi_small = cv2.resize(roi_bgr, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    roi_small = cv2.GaussianBlur(roi_small, (5, 5), 0)

    hsv = cv2.cvtColor(roi_small, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_small, cv2.COLOR_BGR2LAB)
    pixels_hsv = hsv.reshape(-1, 3)
    pixels_lab = lab.reshape(-1, 3).astype(np.float32)

    # Basic masks: saturated & bright enough
    mask = (pixels_hsv[:, 1] >= MIN_SAT) & (pixels_hsv[:, 2] >= MIN_VAL)

    # Remove near-white (low sat but high val) and remove likely-skin hue range
    hue = pixels_hsv[:, 0]
    # skinish hues often sit around 0-30 (red/orange) depending on lighting; be conservative
    skin_mask = (hue > 5) & (hue < 30) & (pixels_hsv[:, 2] > 70) & (pixels_hsv[:, 1] > 30)
    mask = mask & (~skin_mask)

    filtered_lab = pixels_lab[mask]
    filtered_hsv = pixels_hsv[mask]

    if len(filtered_lab) < min_pixels:
        return None, None

    # k-means in LAB (perceptual) space
    n_clusters = min(k, len(filtered_lab))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(filtered_lab)
    counts = np.bincount(labels)
    dom_idx = int(np.argmax(counts))
    dom_lab = km.cluster_centers_[dom_idx]

    # convert lab center to HSV for distance/visualization
    lab_uint8 = np.uint8([[dom_lab]])
    bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0, 0]
    hsv_center = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    return hsv_center, dom_lab


def hsv_distance(a, b, hue_weight=3.0, sv_weight=1.0):
    if a is None or b is None:
        return np.inf
    # a,b are HSV-like arrays (H in [0,180], S,V in [0,255])
    dh = min(abs(int(a[0]) - int(b[0])), 180 - abs(int(a[0]) - int(b[0]))) / 90.0
    ds = abs(int(a[1]) - int(b[1])) / 255.0
    dv = abs(int(a[2]) - int(b[2])) / 255.0
    return hue_weight * dh + sv_weight * (ds + dv)


def is_black_uniform(hsv, black_val_thresh=BLACK_VAL_THRESH, black_sat_thresh=BLACK_SAT_THRESH):
    if hsv is None:
        return False
    return int(hsv[2]) < black_val_thresh and int(hsv[1]) < black_sat_thresh


# ----------------------
# Tracking helpers (very simple, distance-based)
# ----------------------

def match_tracks(prev_tracks, detections, max_dist=80):
    """Match detection centers to previous tracks by greedy nearest neighbor.
    prev_tracks: dict id -> {'center': np.array([x,y]), 'bbox': xyxy}
    detections: list of (center, bbox)

    Returns mapping det_idx -> track_id (or None) and unmatched detections list.
    """
    mapping = [-1] * len(detections)
    used_tracks = set()
    for i, (c, bbox) in enumerate(detections):
        best_id = None
        best_d = max_dist + 1
        for tid, t in prev_tracks.items():
            if tid in used_tracks:
                continue
            d = np.linalg.norm(t['center'] - c)
            if d < best_d and d <= max_dist:
                best_d = d
                best_id = tid
        if best_id is not None:
            mapping[i] = best_id
            used_tracks.add(best_id)
    return mapping


# ----------------------
# Load models
# ----------------------
print('Loading models...')
player_model = YOLO(PLAYER_MODEL_PATH)
ball_model = YOLO(BALL_MODEL_PATH)
print('Models loaded')

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {VIDEO_SOURCE}")

frame_count = 0
learn_lab_samples = []  # list of LAB centers (float)
learn_hsv_samples = []  # corresponding HSV centers (uint8)
team_centroids_lab = None
team_centroids_hsv = None

# For post-lock incremental updates
centroid_counts = None

# Tracking state
tracks = {}  # id -> {'center': np.array, 'bbox': xyxy, 'last_seen': frame_num}
next_track_id = 1
player_histories = defaultdict(list)  # track_id -> list of HSV arrays

last_ball_pts = deque(maxlen=10)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    h, w = frame.shape[:2]
    proximity_radius = int(min(w, h) * PROXIMITY_RADIUS_FRAC)

    # Detect ball
    ball_results = ball_model.predict(source=frame, conf=CONF_BALL, iou=IOU_THRESH, verbose=False)
    ball_xy = None
    if ball_results and len(ball_results[0].boxes) > 0:
        boxes = ball_results[0].boxes
        confs = boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        xyxy = boxes.xyxy.cpu().numpy()[idx]
        ball_xy = bbox_center(xyxy)
        last_ball_pts.append(ball_xy)
    elif len(last_ball_pts) > 0:
        ball_xy = last_ball_pts[-1]

    # Detect players
    player_results = player_model.predict(source=frame, conf=CONF_PLAYER, iou=IOU_THRESH, verbose=False)
    player_boxes = []  # list of (xyxy, conf)
    if player_results and len(player_results[0].boxes) > 0:
        boxes = player_results[0].boxes
        if boxes.cls is not None:
            cls = boxes.cls.cpu().numpy().astype(int)
            keep = cls == 1
        else:
            keep = np.ones(len(boxes), dtype=bool)
        xyxy_all = boxes.xyxy.cpu().numpy()[keep]
        confs = boxes.conf.cpu().numpy()[keep]
        for b, c in zip(xyxy_all, confs):
            player_boxes.append((b, float(c)))

    # Prepare detection centers for tracking
    detections = []  # list of (center, bbox)
    for b, c in player_boxes:
        detections.append((bbox_center(b), b))

    # Match to existing tracks
    mapping = match_tracks(tracks, detections, max_dist=int(min(w, h) * 0.08))

    # Update tracks and assign ids
    det_id_to_track = {}
    for det_idx, (center, bbox) in enumerate(detections):
        tid = mapping[det_idx]
        if tid == -1:
            tid = next_track_id
            next_track_id += 1
        tracks[tid] = {'center': center, 'bbox': bbox, 'last_seen': frame_count}
        det_id_to_track[det_idx] = tid

    # Remove old tracks not seen in last 50 frames
    stale = [tid for tid, t in tracks.items() if frame_count - t['last_seen'] > 50]
    for tid in stale:
        tracks.pop(tid, None)
        player_histories.pop(tid, None)

    # Learning phase: sample torso colors of players near the ball
    if team_centroids_hsv is None and ball_xy is not None:
        samples_this_frame = 0
        for det_idx, (center, bbox) in enumerate(detections):
            if samples_this_frame >= SAMPLES_PER_FRAME_MAX:
                break
            if np.linalg.norm(center - ball_xy) <= proximity_radius:
                roi = bbox_torso_roi(frame, bbox)
                dom_hsv, dom_lab = dominant_color_lab(roi)
                if dom_hsv is not None and dom_lab is not None:
                    # filter out black/uniform
                    if is_black_uniform(dom_hsv):
                        continue
                    learn_lab_samples.append(dom_lab)
                    learn_hsv_samples.append(dom_hsv)
                    samples_this_frame += 1

        # If enough frames & samples, cluster
        if frame_count >= LEARNING_FRAMES and len(learn_lab_samples) >= MIN_LEARN_SAMPLES:
            X = np.array(learn_lab_samples)
            # KMeans in LAB (2 clusters)
            km = KMeans(n_clusters=2, n_init=20, random_state=0)
            labels = km.fit_predict(X)
            centers_lab = km.cluster_centers_
            # Convert labs to HSV for distance and ensure we don't pick a near-black cluster
            centers_hsv = []
            for labc in centers_lab:
                lab_uint8 = np.uint8([[labc]])
                bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0, 0]
                hsvc = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
                centers_hsv.append(hsvc)
            centers_hsv = np.array(centers_hsv)
            centers_lab = np.array(centers_lab)
            # If one center is too dark, drop it and try to pick two non-dark (fallback if both dark)
            val_levels = centers_hsv[:, 2]
            if np.min(val_levels) < 50 and len(centers_lab) == 2:
                # if one is very dark, treat as ref/ignore and attempt to re-cluster by removing dark samples
                non_dark_idx = [i for i in range(len(learn_hsv_samples)) if learn_hsv_samples[i][2] >= 50]
                if len(non_dark_idx) >= MIN_LEARN_SAMPLES:
                    X2 = np.array(learn_lab_samples)[non_dark_idx]
                    km2 = KMeans(n_clusters=2, n_init=20, random_state=0)
                    labels2 = km2.fit_predict(X2)
                    centers_lab = km2.cluster_centers_
                    centers_hsv = []
                    for labc in centers_lab:
                        lab_uint8 = np.uint8([[labc]])
                        bgr = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0, 0]
                        hsvc = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
                        centers_hsv.append(hsvc)
                    centers_hsv = np.array(centers_hsv)
            # finalize
            team_centroids_lab = centers_lab
            team_centroids_hsv = centers_hsv
            centroid_counts = np.ones(len(team_centroids_lab), dtype=np.int32) * 5
            print(f'[INFO] Team centroids locked at frame {frame_count}.')

    # Inference & post-lock updates
    overlay = frame.copy()

    # draw ball
    if ball_xy is not None:
        cv2.circle(overlay, tuple(ball_xy.tolist()), max(6, proximity_radius // 6), (0, 255, 255), -1)
        cv2.circle(overlay, tuple(ball_xy.tolist()), proximity_radius, (0, 255, 255), 2)
        cv2.putText(overlay, 'BALL', (int(ball_xy[0] + 8), int(ball_xy[1] - 8)), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    for det_idx, (center, bbox) in enumerate(detections):
        tid = det_id_to_track[det_idx]
        x1, y1, x2, y2 = map(int, bbox)
        roi = bbox_torso_roi(frame, bbox)
        dom_hsv, dom_lab = dominant_color_lab(roi)

        label = 'Unknown'
        color = (200, 200, 200)

        # reject refs by black uniform
        if dom_hsv is not None and is_black_uniform(dom_hsv):
            label = 'Ref/Ignore'
            color = (60, 60, 60)
        else:
            # temporal smoothing per track
            if dom_hsv is not None:
                hist = player_histories[tid]
                hist.append(dom_hsv.astype(float))
                if len(hist) > SMOOTH_HISTORY:
                    hist.pop(0)
                smooth_hsv = np.mean(np.array(hist), axis=0)
            else:
                smooth_hsv = None

            # assign team if centroids known
            if smooth_hsv is not None and team_centroids_hsv is not None:
                dists = [hsv_distance(smooth_hsv, c) for c in team_centroids_hsv]
                team_id = int(np.argmin(dists))
                label = f'Team {"A" if team_id == 0 else "B"}'
                # visualization color from HSV centroid
                cent_hsv = team_centroids_hsv[team_id].astype(np.uint8)
                bgr = cv2.cvtColor(np.uint8([[cent_hsv]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                color = tuple(int(x) for x in bgr)

                # Post-lock centroid refinement: if this sample is close to assigned centroid, update LAB centroid slightly
                if dom_lab is not None and centroid_counts is not None:
                    # compute distance in HSV
                    dist_to_cent = hsv_distance(smooth_hsv, team_centroids_hsv[team_id])
                    if dist_to_cent < 0.6:  # reasonably close
                        # moving average in LAB space
                        team_centroids_lab[team_id] = (1 - CENTROID_UPDATE_MOMENTUM) * team_centroids_lab[team_id] + CENTROID_UPDATE_MOMENTUM * dom_lab
                        # refresh HSV centroid from updated LAB
                        lab_uint8 = np.uint8([[team_centroids_lab[team_id]]])
                        bgrc = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0, 0]
                        team_centroids_hsv[team_id] = cv2.cvtColor(np.uint8([[bgrc]]), cv2.COLOR_BGR2HSV)[0, 0]
                        centroid_counts[team_id] += 1

        # draw bbox & label
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, DRAW_THICKNESS)
        cv2.putText(overlay, label + f' #{tid}', (x1, max(0, y1 - 8)), FONT, 0.6, color, 2, cv2.LINE_AA)

    # banner
    if team_centroids_hsv is None:
        cv2.rectangle(overlay, (0, 0), (w, 30), (40, 40, 40), -1)
        cv2.putText(overlay, f'Learning colors... frame {frame_count}/{LEARNING_FRAMES} | samples={len(learn_lab_samples)}',
                    (10, 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, (0, 0), (w, 40), (40, 40, 40), -1)
        cv2.putText(overlay, 'Team colors locked (post-refinement running)', (10, 27), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # swatches
        for i, labc in enumerate(team_centroids_lab):
            lab_uint8 = np.uint8([[labc]])
            bgrc = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2BGR)[0, 0]
            hsvc = cv2.cvtColor(np.uint8([[bgrc]]), cv2.COLOR_BGR2HSV)[0, 0]
            bgr_disp = tuple(int(x) for x in bgrc.tolist())
            x0 = 220 + i * 120
            cv2.rectangle(overlay, (x0, 5), (x0 + 50, 35), bgr_disp, -1)
            cv2.putText(overlay, f'Team {"A" if i == 0 else "B"}', (x0 + 56, 27), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Team Color Classifier', overlay)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        print('[INFO] Resetting learning...')
        learn_lab_samples = []
        learn_hsv_samples = []
        team_centroids_lab = None
        team_centroids_hsv = None
        centroid_counts = None
        player_histories.clear()
        tracks.clear()
        next_track_id = 1
        frame_count = 0

cap.release()
cv2.destroyAllWindows()

print('Exiting')
