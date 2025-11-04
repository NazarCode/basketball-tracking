import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import math
import json
import time

class PlayerTracker:
    def __init__(self, yolo_model_path="players_basket.pt"):
        """
        Initialize the player tracker with YOLO model and OCR reader
        """
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize OCR reader (lazy loading for better startup time)
        self.ocr_reader = None
        
        # Player database: {unique_id: {'jersey_number': int, 'jersey_color': tuple, 'last_seen': int}}
        self.players_db = {}
        
        # Tracking history for each detection ID
        self.track_history = defaultdict(list)
        
        # Frame counter
        self.frame_count = 0
        
        # Thresholds
        self.color_similarity_threshold = 60
        self.confidence_threshold = 0.5
        self.person_class_id = 1  # COCO class ID for 'person'
        
        # Performance optimization flags
        self.ocr_every_n_frames = 10  # Run OCR only every N frames
        self.color_every_n_frames = 5   # Extract color every N frames
        self.last_ocr_frame = defaultdict(int)
        self.last_color_frame = defaultdict(int)
        
        # Cache for player features
        self.player_cache = defaultdict(dict)
        
    def init_ocr(self):
        """Lazy initialization of OCR reader"""
        if self.ocr_reader is None:
            print("Initializing OCR reader...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU for stability
    
    def extract_dominant_color(self, image_region):
        """
        Extract dominant color from jersey region (optimized version)
        """
        try:
            # Resize to very small for faster processing
            small_region = cv2.resize(image_region, (64, 64))
            
            # Convert to RGB and get center region (likely jersey area)
            rgb_region = cv2.cvtColor(small_region, cv2.COLOR_BGR2RGB)
            h, w = rgb_region.shape[:2]
            center_region = rgb_region[h//4:3*h//4, w//4:3*w//4]
            
            # Simple average color of center region
            avg_color = np.mean(center_region.reshape(-1, 3), axis=0)
            return tuple(map(int, avg_color))
            
        except Exception as e:
            return (128, 128, 128)  # Default gray color
    
    def extract_jersey_number(self, image_region, track_id):
        """
        Extract jersey number using OCR (optimized with caching)
        """
        try:
            self.init_ocr()
            
            # Focus on upper torso area where numbers are typically located
            h, w = image_region.shape[:2]
            torso_region = image_region[h//6:2*h//3, w//4:3*w//4]
            
            if torso_region.size == 0:
                return None
            
            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
            
            # Simple threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR with strict settings for numbers only
            results = self.ocr_reader.readtext(thresh, 
                                             allowlist='0123456789',
                                             width_ths=0.3,
                                             height_ths=0.3)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    clean_text = ''.join(filter(str.isdigit, text))
                    if clean_text and 1 <= len(clean_text) <= 2:  # Jersey numbers are 1-2 digits
                        number = int(clean_text)
                        if 0 <= number <= 99:
                            return number
            
            return None
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def color_distance(self, color1, color2):
        """Calculate color distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
    
    def find_matching_player(self, jersey_number, jersey_color):
        """Find existing player based on jersey number and color"""
        best_match = None
        best_score = float('inf')
        
        for player_id, player_info in self.players_db.items():
            score = 0
            
            # Jersey number match (most important)
            if jersey_number is not None and player_info['jersey_number'] is not None:
                if player_info['jersey_number'] == jersey_number:
                    score += 0  # Perfect match
                else:
                    score += 1000  # Big penalty for number mismatch
            elif jersey_number is None and player_info['jersey_number'] is None:
                score += 10  # Small penalty for both unknown
            else:
                score += 100  # Medium penalty for partial info
            
            # Color similarity
            color_dist = self.color_distance(jersey_color, player_info['jersey_color'])
            score += color_dist
            
            if score < best_score and score < 200:  # Reasonable threshold
                best_score = score
                best_match = player_id
        
        return best_match
    
    def create_new_player_id(self, jersey_number, jersey_color):
        """Create a new unique player ID"""
        if jersey_number is not None:
            base_id = f"Player#{jersey_number}"
        else:
            base_id = f"Player_Unknown"
        
        # Ensure uniqueness
        counter = 1
        new_id = base_id
        while new_id in self.players_db:
            new_id = f"{base_id}_{counter}"
            counter += 1
        
        # Add to database
        self.players_db[new_id] = {
            'jersey_number': jersey_number,
            'jersey_color': jersey_color,
            'jersey_color_rgb': jersey_color,  # Store for display
            'last_seen': self.frame_count,
            'appearances': 1,
            'first_seen': self.frame_count
        }
        
        return new_id
    
    def process_frame(self, frame):
        """Process a single frame (optimized version)"""
        self.frame_count += 1
        start_time = time.time()
        
        # Run YOLO detection with tracking
        results = self.yolo_model.track(
            frame, 
            persist=True,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            verbose=False  # Reduce console output
        )
        
        annotated_frame = frame.copy()
        player_detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                
                # Crop player region
                player_region = frame[y1:y2, x1:x2]
                
                if player_region.size == 0:
                    continue
                
                jersey_number = None
                jersey_color = (128, 128, 128)  # Default
                
                # Get cached info or extract new features
                cache_key = f"{track_id}"
                
                # Extract color periodically
                if (self.frame_count - self.last_color_frame[track_id] >= self.color_every_n_frames):
                    jersey_color = self.extract_dominant_color(player_region)
                    self.player_cache[cache_key]['color'] = jersey_color
                    self.last_color_frame[track_id] = self.frame_count
                elif 'color' in self.player_cache[cache_key]:
                    jersey_color = self.player_cache[cache_key]['color']
                
                # Extract jersey number less frequently (expensive operation)
                if (self.frame_count - self.last_ocr_frame[track_id] >= self.ocr_every_n_frames):
                    jersey_number = self.extract_jersey_number(player_region, track_id)
                    if jersey_number is not None:
                        self.player_cache[cache_key]['number'] = jersey_number
                    self.last_ocr_frame[track_id] = self.frame_count
                elif 'number' in self.player_cache[cache_key]:
                    jersey_number = self.player_cache[cache_key]['number']
                
                # Find or create player ID
                existing_player_id = self.find_matching_player(jersey_number, jersey_color)
                
                if existing_player_id:
                    player_id = existing_player_id
                    # Update player info
                    self.players_db[player_id]['last_seen'] = self.frame_count
                    self.players_db[player_id]['appearances'] += 1
                    
                    # Update jersey number if we found one and it was previously unknown
                    if (jersey_number is not None and 
                        self.players_db[player_id]['jersey_number'] is None):
                        self.players_db[player_id]['jersey_number'] = jersey_number
                else:
                    player_id = self.create_new_player_id(jersey_number, jersey_color)
                
                # Store detection info
                detection_info = {
                    'player_id': player_id,
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'jersey_number': jersey_number,
                    'jersey_color': jersey_color,
                    'confidence': float(conf)
                }
                player_detections.append(detection_info)
                
                # Draw annotations
                self.draw_player_annotation(annotated_frame, detection_info)
        
        # Add performance info
        process_time = time.time() - start_time
        fps = 1 / process_time if process_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Players: {len(self.players_db)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame, player_detections
    
    def draw_player_annotation(self, frame, detection_info):
        """Draw player annotations on frame"""
        x1, y1, x2, y2 = detection_info['bbox']
        player_id = detection_info['player_id']
        jersey_number = detection_info['jersey_number']
        jersey_color = detection_info['jersey_color']
        confidence = detection_info['confidence']
        
        # Use jersey color for bounding box (BGR format)
        color_bgr = (int(jersey_color[2]), int(jersey_color[1]), int(jersey_color[0]))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
        
        # Prepare label with jersey info
        label_parts = [player_id]
        if jersey_number is not None:
            label_parts.append(f"#{jersey_number}")
        
        # Add color info (RGB values)
        label_parts.append(f"RGB{jersey_color}")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), 
                     color_bgr, -1)
        
        # Draw label text
        cv2.putText(frame, label, 
                   (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (255, 255, 255), 2)
        
        # Draw small color swatch
        cv2.rectangle(frame, (x2 - 30, y1), (x2, y1 + 20), color_bgr, -1)
        cv2.rectangle(frame, (x2 - 30, y1), (x2, y1 + 20), (255, 255, 255), 1)
    
    def get_player_database(self):
        """Get current player database"""
        return self.players_db.copy()
    
    def save_player_database(self, filepath="player_database.json"):
        """Save player database to JSON file with detailed info"""
        # Prepare data for JSON serialization
        save_data = {
            'metadata': {
                'total_frames_processed': self.frame_count,
                'total_players': len(self.players_db),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'players': {}
        }
        
        for player_id, info in self.players_db.items():
            save_data['players'][player_id] = {
                'jersey_number': info['jersey_number'],
                'jersey_color_rgb': info['jersey_color'],
                'first_seen_frame': info['first_seen'],
                'last_seen_frame': info['last_seen'],
                'total_appearances': info['appearances']
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Player database saved to {filepath}")
        return filepath
    
    def load_player_database(self, filepath="player_database.json"):
        """Load player database from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'players' in data:
                for player_id, info in data['players'].items():
                    self.players_db[player_id] = {
                        'jersey_number': info['jersey_number'],
                        'jersey_color': info['jersey_color_rgb'],
                        'jersey_color_rgb': info['jersey_color_rgb'],
                        'first_seen': info['first_seen_frame'],
                        'last_seen': info['last_seen_frame'],
                        'appearances': info['total_appearances']
                    }
            
            print(f"Loaded {len(self.players_db)} players from {filepath}")
            
        except FileNotFoundError:
            print(f"Database file {filepath} not found. Starting fresh.")
        except Exception as e:
            print(f"Error loading database: {e}")
    
    def print_player_summary(self):
        """Print a summary of all detected players"""
        print(f"\n{'='*60}")
        print(f"PLAYER DATABASE SUMMARY (Frame {self.frame_count})")
        print(f"{'='*60}")
        
        if not self.players_db:
            print("No players detected yet.")
            return
        
        for player_id, info in sorted(self.players_db.items()):
            number_str = f"#{info['jersey_number']}" if info['jersey_number'] else "Unknown"
            color_str = f"RGB{info['jersey_color']}"
            
            print(f"Player ID: {player_id}")
            print(f"  Jersey Number: {number_str}")
            print(f"  Jersey Color: {color_str}")
            print(f"  Appearances: {info['appearances']}")
            print(f"  First seen: Frame {info['first_seen']}")
            print(f"  Last seen: Frame {info['last_seen']}")
            print("-" * 40)


def main():
    """Example usage with better performance monitoring"""
    print("Initializing Player Tracker...")
    tracker = PlayerTracker()
    
    # Try to load existing database
    tracker.load_player_database()
    
    # Open video source
    cap = cv2.VideoCapture('test.mp4')  # Change to video file path if needed

    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save player database")
    print("  'p' - Print player summary")
    print("  'c' - Clear player database")
    print("\nStarting tracking...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = tracker.process_frame(frame)
            
            # Display
            cv2.imshow('Player Tracker - Jersey Numbers & Colors', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filepath = tracker.save_player_database()
                print(f"Database saved to {filepath}")
            elif key == ord('p'):
                tracker.print_player_summary()
            elif key == ord('c'):
                tracker.players_db.clear()
                tracker.player_cache.clear()
                print("Player database cleared!")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final summary and save
        print(f"\nSession Summary:")
        print(f"Total frames processed: {tracker.frame_count}")
        print(f"Unique players detected: {len(tracker.players_db)}")
        
        # Auto-save on exit
        if tracker.players_db:
            tracker.save_player_database()
            tracker.print_player_summary()


if __name__ == "__main__":
    main()