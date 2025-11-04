import torch
import cv2
import numpy as np
import time
from collections import deque
import json
import math

# Load models
custom_model = torch.hub.load('ultralytics/yolov5', 'custom',
                             path='/users/nazar/yolov5/runs/train/exp4/weights/best.pt',
                             source='github')
custom_model.conf = 0.15

people_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
people_model.conf = 0.3

print(f"Custom model classes: {custom_model.names}")

class BallTracker:
    def __init__(self, max_disappeared=5, max_distance=80):  # Reduced parameters
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_confidence = 0.7  # Only track high-confidence balls
    
    def register(self, centroid, bbox, score):
        if score < self.min_confidence:  # Only register high-confidence detections
            return
            
        self.objects[self.next_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'score': score,
            'history': deque([centroid], maxlen=15),  # Reduced history
            'age': 0
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        # Filter detections by confidence
        high_conf_detections = [det for det in detections if det['score'] >= self.min_confidence]
        
        if len(high_conf_detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for detection in high_conf_detections:
                self.register(detection['center'], detection['bbox'], detection['score'])
        else:
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            detection_centroids = [det['center'] for det in high_conf_detections]
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                              np.array(detection_centroids), axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = high_conf_detections[col]['center']
                self.objects[object_id]['bbox'] = high_conf_detections[col]['bbox']
                self.objects[object_id]['score'] = high_conf_detections[col]['score']
                self.objects[object_id]['history'].append(high_conf_detections[col]['center'])
                self.objects[object_id]['age'] += 1
                self.disappeared[object_id] = 0
                
                used_row_idxs.add(row)
                used_col_idxs.add(col)
            
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_idxs:
                    self.register(high_conf_detections[col]['center'], 
                                high_conf_detections[col]['bbox'], 
                                high_conf_detections[col]['score'])
        
        # Remove low-confidence or short-lived balls
        to_remove = []
        for object_id, obj_data in self.objects.items():
            if obj_data['age'] < 3 and obj_data['score'] < 0.75:  # Need time to prove legitimacy
                to_remove.append(object_id)
        
        for object_id in to_remove:
            self.deregister(object_id)
        
        return self.objects

class ScoringDetector:
    def __init__(self):
        self.recent_scores = deque(maxlen=20)
        self.score_cooldown = {}
        self.rim_areas = {}
        
    def detect_score(self, ball_tracker, rims, players, frame_time):
        """More lenient scoring detection with better trajectory analysis"""
        scores_detected = []
        
        for rim_idx, (rx1, ry1, rx2, ry2, rim_conf, rim_label) in enumerate(rims):
            rim_center = ((rx1 + rx2) // 2, (ry1 + ry2) // 2)
            rim_width = rx2 - rx1
            rim_height = ry2 - ry1
            rim_area = rim_width * rim_height
            
            # Update rim tracking
            self.rim_areas[rim_idx] = {
                'center': rim_center,
                'width': rim_width,
                'height': rim_height,
                'area': rim_area
            }
            
            # Check cooldown for this rim (reduced cooldown)
            if rim_idx in self.score_cooldown and self.score_cooldown[rim_idx] > 0:
                self.score_cooldown[rim_idx] -= 1
                continue
            
            # Check each tracked ball for scoring
            for ball_id, ball_data in ball_tracker.objects.items():
                # Only check balls that have been tracked for a bit
                if ball_data['age'] < 2:  # Reduced from implicit higher requirement
                    continue
                    
                if self._analyze_scoring_trajectory(ball_data, rim_center, rim_width, rim_height):
                    scorer = self._find_likely_scorer(players, rim_center, ball_data['centroid'])
                    
                    score_data = {
                        'time': frame_time,
                        'timestamp': f"{int(frame_time//60):02d}:{int(frame_time%60):02d}",
                        'rim_position': rim_center,
                        'player': scorer,
                        'ball_id': ball_id,
                        'confidence': self._calculate_score_confidence(ball_data, rim_center, rim_width, rim_height)
                    }
                    
                    scores_detected.append(score_data)
                    self.score_cooldown[rim_idx] = 30  # Reduced from 60 (1 second instead of 2)
                    self.recent_scores.append(score_data)
                    break
        
        return scores_detected
    
    def _analyze_scoring_trajectory(self, ball_data, rim_center, rim_width, rim_height):
        """More lenient trajectory analysis for scoring detection"""
        history = list(ball_data['history'])
        if len(history) < 6:  # Reduced from 10
            return False
        
        rim_x, rim_y = rim_center
        tolerance_x = rim_width * 1.2  # Increased tolerance
        tolerance_y = rim_height * 0.6  # Increased tolerance
        
        # Check for downward motion through rim area
        above_rim_count = 0
        through_rim_count = 0
        below_rim_count = 0
        
        recent_history = history[-12:]  # Look at last 12 positions instead of 15
        
        for i, (bx, by) in enumerate(recent_history):
            # Check horizontal alignment with more tolerance
            if abs(bx - rim_x) <= tolerance_x:
                if by < rim_y - tolerance_y:
                    above_rim_count += 1
                elif rim_y - tolerance_y <= by <= rim_y + tolerance_y:
                    through_rim_count += 1
                elif by > rim_y + tolerance_y:
                    below_rim_count += 1
        
        # More lenient sequence check
        sequence_check = above_rim_count >= 1 and below_rim_count >= 1  # Simplified requirement
        
        # More lenient downward velocity check
        downward_velocity = False
        if len(recent_history) >= 4:  # Reduced from 5
            for i in range(len(recent_history) - 2):  # Look at smaller intervals
                curr_pos = recent_history[i]
                next_pos = recent_history[i + 1]  # Reduced from i + 2
                
                if (abs(curr_pos[0] - rim_x) <= tolerance_x and 
                    abs(next_pos[0] - rim_x) <= tolerance_x):
                    if next_pos[1] > curr_pos[1]:  # Moving downward
                        downward_velocity = True
                        break
        
        # Alternative check: ball was near rim and moved significantly downward
        near_rim_and_dropped = False
        if len(recent_history) >= 6:
            first_half = recent_history[:len(recent_history)//2]
            second_half = recent_history[len(recent_history)//2:]
            
            # Check if ball was near rim in first half
            near_rim_first = any(abs(bx - rim_x) <= tolerance_x and abs(by - rim_y) <= tolerance_y * 2 
                               for bx, by in first_half)
            
            # Check if ball dropped significantly in second half
            if near_rim_first and first_half and second_half:
                avg_y_first = sum(by for bx, by in first_half) / len(first_half)
                avg_y_second = sum(by for bx, by in second_half) / len(second_half)
                y_drop = avg_y_second - avg_y_first
                
                if y_drop > rim_height * 0.5:  # Ball dropped at least half rim height
                    near_rim_and_dropped = True
        
        return sequence_check and (downward_velocity or near_rim_and_dropped)
    
    def _calculate_score_confidence(self, ball_data, rim_center, rim_width, rim_height):
        """More generous confidence calculation for detected scores"""
        history = list(ball_data['history'])
        if len(history) < 3:  # Reduced from 5
            return 0.5  # Higher base confidence
        
        rim_x, rim_y = rim_center
        confidence = 0.6  # Higher base confidence
        
        # Proximity to rim center
        last_pos = history[-1]
        distance_to_rim = math.sqrt((last_pos[0] - rim_x)**2 + (last_pos[1] - rim_y)**2)
        proximity_score = max(0, 1 - distance_to_rim / (rim_width + rim_height))
        confidence += proximity_score * 0.25  # Reduced weight
        
        # Trajectory smoothness (more lenient)
        if len(history) >= 4:  # Reduced requirement
            smoothness = self._calculate_trajectory_smoothness(history[-8:])  # Shorter history
            confidence += smoothness * 0.15  # Reduced weight
        
        return min(1.0, confidence)
    
    def _calculate_trajectory_smoothness(self, positions):
        """Calculate how smooth the trajectory is"""
        if len(positions) < 3:
            return 0
        
        direction_changes = 0
        for i in range(1, len(positions) - 1):
            prev_dir = (positions[i][1] - positions[i-1][1])
            curr_dir = (positions[i+1][1] - positions[i][1])
            
            if prev_dir * curr_dir < 0:  # Direction change
                direction_changes += 1
        
        return max(0, 1 - direction_changes / len(positions))
    
    def _find_likely_scorer(self, players, rim_center, ball_position):
        """Find the most likely scorer based on position and distance"""
        if not players:
            return "Unknown Player"
        
        rim_x, rim_y = rim_center
        ball_x, ball_y = ball_position
        
        # Weight both distance to rim and distance to ball
        best_player = None
        best_score = float('inf')
        
        for i, (px1, py1, px2, py2, conf) in enumerate(players):
            player_center_x = (px1 + px2) // 2
            player_center_y = (py1 + py2) // 2
            
            # Distance to rim (primary factor)
            rim_distance = math.sqrt((player_center_x - rim_x)**2 + (player_center_y - rim_y)**2)
            
            # Distance to ball (secondary factor)
            ball_distance = math.sqrt((player_center_x - ball_x)**2 + (player_center_y - ball_y)**2)
            
            # Combined score (rim distance is more important)
            combined_score = rim_distance * 0.7 + ball_distance * 0.3
            
            if combined_score < best_score:
                best_score = combined_score
                best_player = f"Player {i+1}"
        
        return best_player if best_player else "Unknown Player"

def enhanced_ball_detection(frame, rims, players):
    """More selective ball detection with stricter filtering"""
    balls = []
    
    # Color-based detection with more restrictive parameters
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # More restrictive orange color ranges
    orange_ranges = [
        ([8, 120, 120], [20, 255, 255]),    # Primary orange range
        ([10, 100, 100], [18, 255, 255]),   # Secondary orange range
    ]
    
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for lower, upper in orange_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # More aggressive noise removal
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
    # Fill small gaps
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Additional erosion to reduce false positives
    combined_mask = cv2.erode(combined_mask, kernel_small, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # More restrictive size filtering
        min_area = max(80, frame_area * 0.0001)   # Minimum 80 pixels or 0.01% of frame
        max_area = min(2000, frame_area * 0.002)  # Maximum 2000 pixels or 0.2% of frame
        
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle and fitted ellipse
        x, y, w, h = cv2.boundingRect(contour)
        
        # Stricter aspect ratio check
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10
        if aspect_ratio > 1.6:  # More restrictive
            continue
        
        # Stricter circularity check
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:  # Much more restrictive
            continue
        
        # Solidity check (how "filled" the contour is)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.7:  # Ball should be quite solid
                continue
        
        # Center point
        center_x, center_y = x + w // 2, y + h // 2
        
        # Stricter player occlusion check
        inside_player = False
        player_overlap_threshold = 0.5  # Less overlap allowed
        
        for px1, py1, px2, py2, _ in players:
            overlap_x = max(0, min(x + w, px2) - max(x, px1))
            overlap_y = max(0, min(y + h, py2) - max(y, py1))
            overlap_area = overlap_x * overlap_y
            ball_area_bbox = w * h
            
            if ball_area_bbox > 0 and overlap_area / ball_area_bbox > player_overlap_threshold:
                inside_player = True
                break
        
        if inside_player:
            continue
        
        # Check if detection is near court boundaries (reduce false positives from stands/background)
        border_margin = min(frame_width, frame_height) * 0.05
        if (center_x < border_margin or center_x > frame_width - border_margin or
            center_y < border_margin or center_y > frame_height - border_margin):
            continue
        
        # Enhanced scoring system with stricter thresholds
        size_score = min(1.0, area / (max_area * 0.3))
        circularity_score = min(1.0, circularity * 1.5)
        aspect_ratio_score = max(0, 1 - (aspect_ratio - 1) * 0.8)
        solidity_score = solidity if hull_area > 0 else 0.5
        
        # Color intensity check in HSV
        mask_region = combined_mask[y:y+h, x:x+w]
        color_intensity = np.sum(mask_region) / (w * h * 255)
        color_score = min(1.0, color_intensity * 2)
        
        total_score = (size_score * 0.25 + 
                      circularity_score * 0.3 + 
                      aspect_ratio_score * 0.2 + 
                      solidity_score * 0.15 +
                      color_score * 0.1)
        
        # Only accept high-confidence detections
        if total_score < 0.6:
            continue
        
        balls.append({
            'bbox': (x, y, x + w, y + h),
            'center': (center_x, center_y),
            'area': area,
            'score': total_score,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity if hull_area > 0 else 0
        })
    
    # Sort by score and return only top 2 candidates
    balls.sort(key=lambda x: x['score'], reverse=True)
    
    # Additional filtering: remove balls that are too close to each other
    filtered_balls = []
    min_distance = 50  # Minimum pixels between ball detections
    
    for ball in balls[:3]:  # Check top 3
        is_unique = True
        for existing_ball in filtered_balls:
            distance = math.sqrt((ball['center'][0] - existing_ball['center'][0])**2 + 
                               (ball['center'][1] - existing_ball['center'][1])**2)
            if distance < min_distance:
                is_unique = False
                break
        
        if is_unique:
            filtered_balls.append(ball)
    
    return filtered_balls[:2]  # Return maximum 2 balls

# Initialize components
ball_tracker = BallTracker(max_disappeared=15, max_distance=150)
scoring_detector = ScoringDetector()

# Video setup
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps} FPS, {total_frames} frames")

# Tracking variables
scores = []
frame_count = 0
start_time = time.time()
process_every_n_frames = 1  # Process every frame for better tracking

# Performance optimization
last_detections = {'rims': [], 'players': []}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = frame_count / fps
    
    # Rim detection (every frame for accurate tracking)
    rim_results = custom_model(frame, size=640)
    rims = []
    
    for *xyxy, conf, cls in rim_results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        label = custom_model.names[int(cls)]
        confidence = float(conf)
        
        if label.lower() in ["hoop", "rim", "basket"] and confidence > 0.15:
            rims.append((x1, y1, x2, y2, confidence, label))
    
    # Player detection (every 2 frames to balance accuracy and performance)
    if frame_count % 2 == 0:
        people_results = people_model(frame, size=640)
        players = []
        
        for *xyxy, conf, cls in people_results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = people_model.names[int(cls)]
            confidence = float(conf)
            
            if label == "person" and confidence > 0.4:
                players.append((x1, y1, x2, y2, confidence))
        
        last_detections['players'] = players
    else:
        players = last_detections['players']
    
    # Ball detection and tracking
    ball_detections = enhanced_ball_detection(frame, rims, players)
    tracked_balls = ball_tracker.update(ball_detections)
    
    # Score detection
    new_scores = scoring_detector.detect_score(ball_tracker, rims, players, current_time)
    
    for score_event in new_scores:
        scores.append(score_event)
        print(f"\n🏀 SCORE DETECTED! 🏀")
        print(f"Time: {score_event['timestamp']}")
        print(f"Player: {score_event['player']}")
        print(f"Confidence: {score_event['confidence']:.2f}")
        
        # Save scores to file
        with open('basketball_scores.json', 'w') as f:
            json.dump(scores, f, indent=2, default=str)
    
    # VISUALIZATION
    # Draw rims
    for x1, y1, x2, y2, conf, label in rims:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw players
    for i, (x1, y1, x2, y2, conf) in enumerate(players):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, f"P{i+1}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw tracked balls with trajectory (only show stable tracks)
    for ball_id, ball_data in tracked_balls.items():
        # Only show balls that have been tracked for a few frames
        if ball_data['age'] < 3:
            continue
            
        x1, y1, x2, y2 = ball_data['bbox']
        score = ball_data['score']
        
        # Color based on confidence
        if score > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif score > 0.7:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (255, 0, 255)  # Magenta - low confidence
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Ball-{ball_id} {score:.2f}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw trajectory (shorter trail)
        history = list(ball_data['history'])
        if len(history) > 3:  # Only show if we have enough points
            for i in range(max(1, len(history)-8), len(history)):  # Last 8 points only
                alpha = i / len(history)  # Fade older points
                point_color = tuple(int(c * alpha) for c in color)
                if i > 0:
                    cv2.line(frame, history[i-1], history[i], point_color, 2)
    
    # Status information
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    timestamp = f"{minutes:02d}:{seconds:02d}"
    
    # Background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (500, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    cv2.putText(frame, f"Time: {timestamp} | Frame: {frame_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Scores: {len(scores)} | Tracked Balls: {len(tracked_balls)}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Rims: {len(rims)} | Players: {len(players)}", 
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show recent score
    if new_scores:
        for score_event in new_scores:
            cv2.putText(frame, f"🏀 SCORE: {score_event['player']} 🏀", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    elif scores:
        last_score = scores[-1]
        cv2.putText(frame, f"Last: {last_score['player']} at {last_score['timestamp']}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    
    cv2.imshow("Enhanced Basketball Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup and final statistics
cap.release()
cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time
avg_fps = frame_count / total_time

print(f"\n📊 FINAL STATISTICS 📊")
print(f"Processed {frame_count} frames in {total_time:.1f} seconds")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Total scores detected: {len(scores)}")

if scores:
    print(f"\n🏀 SCORING SUMMARY 🏀")
    for i, score in enumerate(scores):
        print(f"{i+1}. {score['timestamp']} - {score['player']} (conf: {score['confidence']:.2f})")

print(f"\nResults saved to: basketball_scores.json")