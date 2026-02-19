"""ZED 2i Camera Preview (UVC mode on macOS)

Since ZED SDK doesn't support macOS, this uses the camera as a
standard UVC device via OpenCV. The raw feed is a side-by-side
stereo image (left + right eye).

Controls:
  q     — quit
  s     — save screenshot
  1     — show side-by-side (default)
  2     — show left eye only
  3     — show right eye only
"""

import cv2
import sys
import time
import numpy as np
from datetime import datetime


def find_zed_camera(max_index=5):
    """Try to find the ZED camera by checking video indices."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  Camera index {i}: {int(w)}x{int(h)}")
            # ZED 2i side-by-side typically outputs at 2560x720 or wider
            if w >= 2560:
                print(f"  → ZED 2i found at index {i}")
                return cap
            cap.release()
    return None


def main():
    print("Searching for ZED 2i camera...")
    cap = find_zed_camera()

    if cap is None:
        # Fallback: try index 0
        print("No wide stereo camera found. Trying index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open any camera.")
            sys.exit(1)

    # Try to set higher resolution (ZED 2i supports up to 2K stereo)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Capture resolution: {w}x{h} @ {fps:.0f} fps")

    mode = 1  # 1=side-by-side, 2=left, 3=right
    mode_names = {1: "Side-by-Side", 2: "Left Eye", 3: "Right Eye"}

    print("\nControls: q=quit, s=screenshot, 1=stereo, 2=left, 3=right")
    print("Starting preview...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        half_w = w // 2

        if mode == 2:
            display = frame[:, :half_w]
        elif mode == 3:
            display = frame[:, half_w:]
        else:
            display = frame

        # Add mode label
        cv2.putText(display, mode_names[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ZED 2i Preview", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"zed_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
