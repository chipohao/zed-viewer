"""ZED 2i Motion Tracker for macOS (no SDK required)

Features:
  - Live stereo camera feed with factory calibration
  - Stereo depth map (disparity via StereoSGBM)
  - Body pose tracking (MediaPipe, 33 landmarks)
  - Dense optical flow visualization

Controls:
  q       — quit
  s       — save screenshot
  1       — left eye + pose overlay (default)
  2       — stereo depth map
  3       — optical flow visualization
  4       — side-by-side: left + depth
  d       — toggle depth colormap
  +/-     — adjust depth sensitivity
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import configparser
import sys
import os
import time
from datetime import datetime

# ── Calibration ──────────────────────────────────────────────

CALIB_FILE = os.path.join(os.path.dirname(__file__), "SN36291917.conf")
CAMERA_INDEX = 1
RESOLUTION = "HD"  # HD = 1280x720 per eye


def load_calibration(conf_path, resolution="HD"):
    """Load ZED factory calibration and build rectification maps."""
    config = configparser.ConfigParser()
    config.read(conf_path)

    res = resolution
    left_sec = f"LEFT_CAM_{res}"
    right_sec = f"RIGHT_CAM_{res}"

    # Image size per eye
    sizes = {"2K": (2208, 1242), "FHD": (1920, 1080), "HD": (1280, 720), "VGA": (672, 376)}
    w, h = sizes[res]
    image_size = (w, h)

    # Left camera intrinsics
    K_left = np.array([
        [float(config[left_sec]["fx"]), 0, float(config[left_sec]["cx"])],
        [0, float(config[left_sec]["fy"]), float(config[left_sec]["cy"])],
        [0, 0, 1]
    ])
    D_left = np.array([
        float(config[left_sec]["k1"]),
        float(config[left_sec]["k2"]),
        float(config[left_sec]["p1"]),
        float(config[left_sec]["p2"]),
        float(config[left_sec]["k3"])
    ])

    # Right camera intrinsics
    K_right = np.array([
        [float(config[right_sec]["fx"]), 0, float(config[right_sec]["cx"])],
        [0, float(config[right_sec]["fy"]), float(config[right_sec]["cy"])],
        [0, 0, 1]
    ])
    D_right = np.array([
        float(config[right_sec]["k1"]),
        float(config[right_sec]["k2"]),
        float(config[right_sec]["p1"]),
        float(config[right_sec]["p2"]),
        float(config[right_sec]["k3"])
    ])

    # Stereo parameters
    baseline = float(config["STEREO"]["Baseline"]) / 1000.0  # mm → m
    RX = float(config["STEREO"][f"RX_{res}"])
    CV = float(config["STEREO"][f"CV_{res}"])
    RZ = float(config["STEREO"][f"RZ_{res}"])
    TY = float(config["STEREO"].get("TY", "0"))
    TZ = float(config["STEREO"].get("TZ", "0"))

    # Rotation matrix from Rodrigues angles
    R = cv2.Rodrigues(np.array([RX, CV, RZ]))[0]
    # Translation vector
    T = np.array([-baseline, TY / 1000.0, TZ / 1000.0])

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left, D_left, K_right, D_right,
        image_size, R, T,
        alpha=0, flags=0
    )

    # Rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2, image_size, cv2.CV_32FC1
    )

    focal = P1[0, 0]
    print(f"Calibration loaded: {res} ({w}x{h}), baseline={baseline*1000:.1f}mm, focal={focal:.1f}px")

    return map_left_x, map_left_y, map_right_x, map_right_y, Q, focal, baseline


# ── Stereo Depth ─────────────────────────────────────────────

def create_stereo_matcher(num_disp=128, block_size=7):
    """Create StereoSGBM matcher with WLS filter."""
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)
    return left_matcher, right_matcher, wls_filter


def compute_depth(left_rect, right_rect, left_matcher, right_matcher, wls_filter):
    """Compute filtered disparity map."""
    gray_l = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disp_l = left_matcher.compute(gray_l, gray_r)
    disp_r = right_matcher.compute(gray_r, gray_l)

    filtered = wls_filter.filter(disp_l, gray_l, None, disp_r)
    # Normalize for display
    filtered_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return filtered_norm


# ── Pose Tracking (MediaPipe Tasks API) ──────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

# Pose landmark connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),   # left eye
    (0, 4), (4, 5), (5, 6), (6, 8),   # right eye
    (9, 10),                            # mouth
    (11, 12),                           # shoulders
    (11, 13), (13, 15),                 # left arm
    (12, 14), (14, 16),                 # right arm
    (15, 17), (15, 19), (15, 21),      # left hand
    (16, 18), (16, 20), (16, 22),      # right hand
    (11, 23), (12, 24),                 # torso
    (23, 24),                           # hips
    (23, 25), (25, 27),                 # left leg
    (24, 26), (26, 28),                 # right leg
    (27, 29), (29, 31), (27, 31),      # left foot
    (28, 30), (30, 32), (28, 32),      # right foot
]


def setup_pose():
    """Initialize MediaPipe Pose Landmarker (Tasks API)."""
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return landmarker


def draw_pose(image, landmarker, timestamp_ms):
    """Run pose detection and draw landmarks on image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = image.shape[:2]
    if results.pose_landmarks:
        for landmarks in results.pose_landmarks:
            # Draw connections
            for start_idx, end_idx in POSE_CONNECTIONS:
                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                if pt1.visibility > 0.5 and pt2.visibility > 0.5:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw landmarks
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                if lm.visibility > 0.5:
                    cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    return image, results


# ── Optical Flow ─────────────────────────────────────────────

def compute_optical_flow(prev_gray, curr_gray):
    """Dense optical flow (Farneback) → HSV color visualization."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*curr_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # hue = direction
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value = magnitude
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=== ZED 2i Motion Tracker (macOS) ===\n")

    # Load calibration
    if not os.path.exists(CALIB_FILE):
        print(f"ERROR: Calibration file not found: {CALIB_FILE}")
        sys.exit(1)
    map_lx, map_ly, map_rx, map_ry, Q, focal, baseline = load_calibration(CALIB_FILE, RESOLUTION)

    # Open camera
    print(f"Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        sys.exit(1)

    # Set resolution (2560x720 for HD stereo side-by-side)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Capture: {actual_w}x{actual_h} @ {actual_fps:.0f} fps")

    # Setup components
    left_matcher, right_matcher, wls_filter = create_stereo_matcher()
    landmarker = setup_pose()
    start_time = time.monotonic()

    mode = 1  # 1=pose, 2=depth, 3=optical flow, 4=split view
    mode_names = {
        1: "Pose Tracking",
        2: "Depth Map",
        3: "Optical Flow",
        4: "Left + Depth"
    }
    depth_cmap = cv2.COLORMAP_MAGMA
    cmaps = [cv2.COLORMAP_MAGMA, cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO,
             cv2.COLORMAP_TURBO, cv2.COLORMAP_PLASMA]
    cmap_names = ["MAGMA", "JET", "INFERNO", "TURBO", "PLASMA"]
    cmap_idx = 0
    num_disp = 128

    prev_gray = None
    frame_count = 0

    print(f"\nControls:")
    print(f"  1 = Pose Tracking    2 = Depth Map")
    print(f"  3 = Optical Flow     4 = Left + Depth")
    print(f"  d = cycle colormap   +/- = depth range")
    print(f"  s = screenshot       q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        half_w = actual_w // 2

        # Split stereo pair
        raw_left = frame[:, :half_w]
        raw_right = frame[:, half_w:]

        # Rectify
        left_rect = cv2.remap(raw_left, map_lx, map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(raw_right, map_rx, map_ry, cv2.INTER_LINEAR)

        # ── Mode 1: Pose Tracking ──
        if mode == 1:
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            display, results = draw_pose(left_rect.copy(), landmarker, timestamp_ms)

        # ── Mode 2: Depth Map ──
        elif mode == 2:
            disp = compute_depth(left_rect, right_rect, left_matcher, right_matcher, wls_filter)
            display = cv2.applyColorMap(disp, depth_cmap)

        # ── Mode 3: Optical Flow ──
        elif mode == 3:
            curr_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                display = compute_optical_flow(prev_gray, curr_gray)
            else:
                display = left_rect.copy()
            prev_gray = curr_gray

        # ── Mode 4: Split View (Left + Depth) ──
        elif mode == 4:
            disp = compute_depth(left_rect, right_rect, left_matcher, right_matcher, wls_filter)
            depth_color = cv2.applyColorMap(disp, depth_cmap)
            # Resize both to half width for side-by-side
            h = left_rect.shape[0]
            hw = left_rect.shape[1] // 2
            left_small = cv2.resize(left_rect, (hw, h // 2))
            depth_small = cv2.resize(depth_color, (hw, h // 2))
            display = np.hstack([left_small, depth_small])

        # HUD overlay
        label = f"{mode_names[mode]}"
        if mode in (2, 4):
            label += f" | {cmap_names[cmap_idx]} | disp={num_disp}"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {actual_fps:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ZED 2i Motion Tracker", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"zed_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = os.path.join(os.path.dirname(__file__), fname)
            cv2.imwrite(path, display)
            print(f"Saved: {path}")
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
            prev_gray = None
        elif key == ord('4'):
            mode = 4
        elif key == ord('d'):
            cmap_idx = (cmap_idx + 1) % len(cmaps)
            depth_cmap = cmaps[cmap_idx]
            print(f"Colormap: {cmap_names[cmap_idx]}")
        elif key == ord('+') or key == ord('='):
            num_disp = min(256, num_disp + 16)
            left_matcher, right_matcher, wls_filter = create_stereo_matcher(num_disp=num_disp)
            print(f"numDisparities: {num_disp}")
        elif key == ord('-'):
            num_disp = max(16, num_disp - 16)
            left_matcher, right_matcher, wls_filter = create_stereo_matcher(num_disp=num_disp)
            print(f"numDisparities: {num_disp}")

        frame_count += 1

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
