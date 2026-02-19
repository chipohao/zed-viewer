"""Diagnose camera access on macOS."""
import cv2
import sys

print(f"OpenCV version: {cv2.__version__}")
print(f"Python: {sys.version}")
print(f"OpenCV build info (videoio):")
info = cv2.getBuildInformation()
for line in info.split('\n'):
    if any(k in line.lower() for k in ['video', 'avfoundation', 'backend']):
        print(f"  {line.strip()}")

print("\n--- Scanning camera indices 0-5 ---")
for i in range(6):
    print(f"\nTrying index {i}...", end=" ")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        backend = cap.getBackendName()
        ret, frame = cap.read()
        print(f"OPEN — {w}x{h} @ {fps:.0f}fps, backend={backend}, read={'OK' if ret else 'FAIL'}")
        if ret:
            print(f"    Frame shape: {frame.shape}, dtype: {frame.dtype}")
        cap.release()
    else:
        print("cannot open")

print("\n--- Trying AVFoundation backend explicitly ---")
for i in range(6):
    print(f"\nTrying AVFoundation index {i}...", end=" ")
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        print(f"OPEN — {w}x{h}, read={'OK' if ret else 'FAIL'}")
        if ret:
            print(f"    Frame shape: {frame.shape}")
        cap.release()
    else:
        print("cannot open")

print("\nDone.")
