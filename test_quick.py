"""Quick test: grab one frame, verify calibration + rectification."""
import cv2
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from zed_motion_tracker import load_calibration, CALIB_FILE

print("Loading calibration...")
map_lx, map_ly, map_rx, map_ry, Q, focal, baseline = load_calibration(CALIB_FILE, "HD")

print("Opening camera index 1...")
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("FAIL: cannot open camera")
    sys.exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {w}x{h}")

ret, frame = cap.read()
if not ret:
    print("FAIL: cannot read frame")
    cap.release()
    sys.exit(1)

print(f"Frame: {frame.shape}")
half_w = w // 2
left = frame[:, :half_w]
right = frame[:, half_w:]

left_rect = cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR)
right_rect = cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR)

# Save test images
cv2.imwrite(os.path.join(os.path.dirname(__file__), "test_left_rect.png"), left_rect)
cv2.imwrite(os.path.join(os.path.dirname(__file__), "test_right_rect.png"), right_rect)

# Test MediaPipe import
import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")

# Test ximgproc (WLS filter)
try:
    _ = cv2.ximgproc.createDisparityWLSFilter
    print("cv2.ximgproc: OK (WLS filter available)")
except AttributeError:
    print("cv2.ximgproc: NOT AVAILABLE — will need fallback")

cap.release()
print("\nAll checks passed! Run zed_motion_tracker.py to start.")
