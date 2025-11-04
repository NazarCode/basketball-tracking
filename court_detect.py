import cv2
import numpy as np

def isolate_court_floor(image):
    # Convert to HSV to isolate wood floor colors
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for typical wood color (adjust as needed)
    lower = np.array([10, 50, 50])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest

def detect_paint_area(image, court_mask):
    # The paint/key is often a large rectangle near the basket.
    # We try to find rectangles inside the court.
    contours, _ = cv2.findContours(court_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    paint_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue  # too small
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            # This could be paint - heuristic based on size/shape
            paint_contour = approx
            break
    return paint_contour

def detect_three_point_line(image, court_mask):
    # Use Canny + contours to detect arcs
    edges = cv2.Canny(court_mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arcs = []
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 100:
            continue
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        # Look for curved shapes (more than 5 points)
        if len(approx) > 8:
            arcs.append(cnt)
    
    # Choose largest arc as 3pt line candidate
    if arcs:
        three_pt_line = max(arcs, key=cv2.contourArea)
        return three_pt_line
    return None

def draw_court_features(image, court_contour, paint_contour, three_point_contour):
    output = image.copy()

    # Draw court boundary
    if court_contour is not None:
        cv2.drawContours(output, [court_contour], -1, (0,255,0), 3)

    # Draw paint/key area
    if paint_contour is not None:
        cv2.drawContours(output, [paint_contour], -1, (255,0,0), 3)

    # Draw three point line
    if three_point_contour is not None:
        cv2.drawContours(output, [three_point_contour], -1, (0,0,255), 3)

    return output

def detect_basketball_court(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not load image.")
        return

    court_mask = isolate_court_floor(img)
    court_contour = find_largest_contour(court_mask)
    
    if court_contour is None:
        print("Could not detect court boundary.")
        return
    
    # Create a mask for just the court
    court_only_mask = np.zeros_like(court_mask)
    cv2.drawContours(court_only_mask, [court_contour], -1, 255, thickness=cv2.FILLED)

    paint_contour = detect_paint_area(img, court_only_mask)
    three_point_contour = detect_three_point_line(img, court_only_mask)

    output = draw_court_features(img, court_contour, paint_contour, three_point_contour)

    cv2.imshow("Detected Basketball Court", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
detect_basketball_court("test.webp")
