#!/usr/bin/env python3
"""
Court outline from a single frame, rim-assisted (color agnostic).
------------------------------------------------------------

- Works even in multi-sport gyms by using rim bbox to pick the correct baseline.
- Color agnostic: detects edges in grayscale, so any line color works.
- Uses single-view metrology (vanishing points) for metric rectification.
- Fits a FIBA 15m x 28m court polygon and warps it back to original frame.

Usage:
------
python outline_court.py --image path/to/frame.jpg --out overlay.png --standard fiba --rim x y w h
# add --debug to save intermediate results
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ---------------------------
# Data structures & helpers
# ---------------------------

@dataclass
class LineSeg:
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    angle: float  # radians in [0, pi)
    length: float

def _homog_point(x, y) -> np.ndarray:
    return np.array([x, y, 1.0], dtype=np.float64)

def _line_from_points(p1, p2) -> np.ndarray:
    if p1 is None or p2 is None:
        raise ValueError("Cannot form line from None point.")
    P1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    P2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
    l = np.cross(P1, P2)
    if np.allclose(l, 0):
        raise ValueError("Degenerate line from identical points.")
    return l / np.linalg.norm(l[:2] + 1e-9)


def _intersection(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p / p[2]

def _apply_H_pts(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    warped = (H @ pts_h.T).T
    warped /= (warped[:, 2:3] + 1e-9)
    return warped[:, :2]

def _warp_size(img_shape, H):
    h, w = img_shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float64)
    wc = _apply_H_pts(corners, H)
    xmin, ymin = wc.min(axis=0)
    xmax, ymax = wc.max(axis=0)
    size = (int(math.ceil(xmax - xmin)), int(math.ceil(ymax - ymin)))
    offset = np.array([-xmin, -ymin])
    Ht = np.array([[1,0,offset[0]],[0,1,offset[1]],[0,0,1]], dtype=np.float64)
    return size, Ht @ H

def _normalize_angle(theta: float) -> float:
    theta = theta % math.pi
    if theta < 0:
        theta += math.pi
    return theta

# ---------------------------
# Detection & grouping
# ---------------------------

def detect_line_segments(img_bgr: np.ndarray) -> List[LineSeg]:
    """Detect lines in grayscale (color agnostic)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                             minLineLength=200, maxLineGap=20)

    segs: List[LineSeg] = []
    if linesP is not None:
        for x1,y1,x2,y2 in linesP[:,0,:]:
            length = math.hypot(x2-x1, y2-y1)
            if length < 80:
                continue
            angle = _normalize_angle(math.atan2(y2-y1, x2-x1))
            segs.append(LineSeg((x1,y1),(x2,y2), angle, length))
    return segs

def cluster_directions(segs: List[LineSeg]) -> Tuple[List[LineSeg], List[LineSeg]]:
    """Cluster into 2 direction groups using simple 2-means on angles."""
    if len(segs) < 4:
        return segs, []
    X = np.array([[math.cos(2*s.angle), math.sin(2*s.angle)] for s in segs], dtype=np.float64)
    idxs = np.arange(len(segs))
    c1 = X[0]
    dists = np.linalg.norm(X - c1, axis=1)
    c2 = X[dists.argmax()]
    for _ in range(15):
        d1 = np.linalg.norm(X - c1, axis=1)
        d2 = np.linalg.norm(X - c2, axis=1)
        g1 = idxs[d1 <= d2]
        g2 = idxs[d1 > d2]
        if len(g1)==0 or len(g2)==0:
            break
        c1 = X[g1].mean(axis=0)
        c2 = X[g2].mean(axis=0)
    G1 = [segs[i] for i in g1.tolist()]
    G2 = [segs[i] for i in g2.tolist()]
    a1 = np.mean([s.angle for s in G1]) if G1 else 0.0
    a2 = np.mean([s.angle for s in G2]) if G2 else 0.0
    if a1 > a2:
        G1, G2 = G2, G1
    return G1, G2

def vanishing_point_from_segments(segs: List[LineSeg]) -> Optional[np.ndarray]:
    if len(segs) < 2:
        return None
    L = []
    for s in segs:
        l = _line_from_points(s.p1, s.p2)
        L.append(l)
    L = np.stack(L, axis=0)
    U, S, Vt = np.linalg.svd(L)
    v = Vt[-1, :]
    if abs(v[2]) < 1e-12:
        return None
    return v / v[2]

# ---------------------------
# Rectification
# ---------------------------

def affine_rectify(img: np.ndarray, v1: np.ndarray, v2: np.ndarray):
    l = np.cross(v1, v2)
    H_a = np.array([[1,0,0],[0,1,0],[l[0], l[1], l[2]]], dtype=np.float64)
    size, Htot = _warp_size(img.shape, H_a)
    rect = cv2.warpPerspective(img, Htot, size, flags=cv2.INTER_LINEAR)
    return rect, Htot

def _direction_from_rectified_segments(segs: List[LineSeg], H: np.ndarray) -> np.ndarray:
    dirs = []
    for s in segs:
        pts = np.array([[s.p1[0], s.p1[1]], [s.p2[0], s.p2[1]]], dtype=np.float64)
        w = _apply_H_pts(pts, H)
        d = w[1] - w[0]
        nrm = np.linalg.norm(d)
        if nrm > 1e-6:
            dirs.append(d / nrm)
    if not dirs:
        return np.array([1.0, 0.0])
    dmean = np.mean(dirs, axis=0)
    nrm = np.linalg.norm(dmean)
    return dmean / (nrm + 1e-9)

def metric_rectify(rect_img: np.ndarray, H_affine: np.ndarray,
                   g1_segs: List[LineSeg], g2_segs: List[LineSeg]):
    u = _direction_from_rectified_segments(g1_segs, H_affine)
    v = _direction_from_rectified_segments(g2_segs, H_affine)
    B = np.column_stack([u, v])
    if abs(np.linalg.det(B)) < 1e-6:
        M = np.eye(2)
    else:
        M = np.linalg.inv(B)
    Hm = np.array([[M[0,0], M[0,1], 0.0],
                   [M[1,0], M[1,1], 0.0],
                   [0.0,    0.0,    1.0]], dtype=np.float64)
    Htot = Hm @ H_affine
    size, Htot2 = _warp_size(rect_img.shape, Hm)
    rect2 = cv2.warpPerspective(rect_img, Hm, size, flags=cv2.INTER_LINEAR)
    return rect2, Htot2

# ---------------------------
# Baseline & sideline selection with rim bias
# ---------------------------

def fit_lines_with_angle(rect_img: np.ndarray, target_angle: float, tol_deg: float) -> List[np.ndarray]:
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=160,
                             minLineLength=300, maxLineGap=30)
    if linesP is None:
        return []
    target = math.radians(target_angle)
    found = []
    for x1,y1,x2,y2 in linesP[:,0,:]:
        ang = _normalize_angle(math.atan2(y2-y1, x2-x1))
        d = min(abs(ang - target), abs(ang - (target + math.pi)), abs(ang - (target - math.pi)))
        if d <= math.radians(tol_deg):
            L = _line_from_points((x1,y1),(x2,y2))
            found.append(L)
    return found

def line_center_distance_to_rim(line: np.ndarray, H_inv: np.ndarray, rim_cx: float, rim_cy: float) -> float:
    w, h = 1920, 1080  # just large bounds; adjust if needed
    pts = line_endpoints_in_image(line, w, h)
    if pts is None:
        return float('inf')
    pts_orig = _apply_H_pts(pts, H_inv)
    mid = pts_orig.mean(axis=0)
    return np.linalg.norm(mid - np.array([rim_cx, rim_cy]))

def line_endpoints_in_image(l: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
    borders = [
        np.array([1, 0, 0], dtype=np.float64),
        np.array([1, 0, -w], dtype=np.float64),
        np.array([0, 1, 0], dtype=np.float64),
        np.array([0, 1, -h], dtype=np.float64),
    ]
    pts = []
    for b in borders:
        p = _intersection(l, b)
        if p is None: continue
        x, y = p[0], p[1]
        if -1 <= x <= w+1 and -1 <= y <= h+1:
            pts.append([x, y])
    if len(pts) < 2:
        return None
    pts = np.array(pts, dtype=np.float64)
    dmax = 0
    best_pair = (0,1)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i]-pts[j])
            if d > dmax:
                dmax = d
                best_pair = (i,j)
    return pts[list(best_pair)]

# ---------------------------
# Court polygon placement
# ---------------------------

def place_court_polygon(rect_img: np.ndarray,
                        baseline_line: np.ndarray,
                        sideline_line: np.ndarray,
                        standard: str = "fiba") -> np.ndarray:
    if standard.lower() == "nba":
        width_m, length_m = 15.24, 28.65
    else:
        width_m, length_m = 15.0, 28.0
    h, w = rect_img.shape[:2]
    be = line_endpoints_in_image(baseline_line, w, h)
    se = line_endpoints_in_image(sideline_line, w, h)
    if be is None or se is None:
        raise RuntimeError("Could not get line endpoints.")
    corner = _intersection(baseline_line, sideline_line)
    if corner is None:
        raise RuntimeError("Baseline and sideline don't intersect.")
    C = corner[:2]
    u = be[1] - be[0]; u /= np.linalg.norm(u)+1e-9
    v = se[1] - se[0]; v /= np.linalg.norm(v)+1e-9
    baseline_len_px = np.linalg.norm(be[1] - be[0])
    ppm_u = baseline_len_px / width_m
    ppm_v = ppm_u
    Wpx = width_m * ppm_u
    Lpx = length_m * ppm_v
    P1 = C
    P2 = C + u * Wpx
    P3a = P2 + v * Lpx
    P4a = C + v * Lpx
    P3b = P2 - v * Lpx
    P4b = C - v * Lpx
    def in_bounds(poly):
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return (min(xs) > -w*0.5 and max(xs) < w*1.5 and min(ys) > -h*0.5 and max(ys) < h*1.5)
    poly_a = np.array([P1, P2, P3a, P4a], dtype=np.float64)
    poly_b = np.array([P1, P2, P3b, P4b], dtype=np.float64)
    return poly_a if in_bounds(poly_a) or not in_bounds(poly_b) else poly_b

# ---------------------------
# Overlay
# ---------------------------

def draw_polygon(img: np.ndarray, poly: np.ndarray, color=(0,255,0), thickness=3) -> np.ndarray:
    out = img.copy()
    pts = poly.reshape(-1,1,2).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--standard", choices=["fiba","nba"], default="fiba")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--rim", type=int, nargs=4, metavar=("x","y","w","h"))
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    segs = detect_line_segments(img)
    if len(segs) < 4:
        print("Not enough segments.")
        sys.exit(2)

    G1, G2 = cluster_directions(segs)
    v1 = vanishing_point_from_segments(G1)
    v2 = vanishing_point_from_segments(G2)
    if v1 is None or v2 is None:
        print("Could not find two vanishing points — check image/lines.")
        sys.exit(1)

    rect1, H_aff = affine_rectify(img, v1, v2)
    rect2, H_tot = metric_rectify(rect1, H_aff, G1, G2)

    # Fit baselines & sidelines
    baselines = fit_lines_with_angle(rect2, 0.0, tol_deg=20.0)
    sidelines = fit_lines_with_angle(rect2, 90.0, tol_deg=20.0)
    if not baselines or not sidelines:
        print("No baseline/sideline candidates.")
        sys.exit(3)

    H_inv = np.linalg.inv(H_tot)
    if args.rim:
        rim_cx = args.rim[0] + args.rim[2]/2
        rim_cy = args.rim[1] + args.rim[3]/2
        baselines.sort(key=lambda L: line_center_distance_to_rim(L, H_inv, rim_cx, rim_cy))
    baseline = baselines[0]
    sideline = sidelines[0]  # could add dimension check here

    poly_rect = place_court_polygon(rect2, baseline, sideline, standard=args.standard)
    poly_orig = _apply_H_pts(poly_rect, H_inv)
    overlay = draw_polygon(img, poly_orig)
    cv2.imwrite(args.out, overlay)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
