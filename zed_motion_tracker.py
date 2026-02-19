"""ZED 2i Motion Tracker for macOS (no SDK required)

Features:
  - Live stereo camera feed with factory calibration
  - Stereo depth map (disparity via StereoSGBM)
  - Body pose tracking (MediaPipe, 33 landmarks)
  - Dense optical flow visualization
  - OSC output for TouchDesigner / Ableton Live
  - Syphon video sharing (macOS)
  - Virtual webcam output (via OBS Virtual Camera)

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
import argparse
from datetime import datetime

from pythonosc.udp_client import SimpleUDPClient

try:
    import syphon
    from syphon.utils.numpy import copy_image_to_mtl_texture
    from syphon.utils.raw import create_mtl_texture
    HAS_SYPHON = True
except ImportError:
    HAS_SYPHON = False

try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False

# ── Calibration ──────────────────────────────────────────────

CALIB_FILE = os.path.join(os.path.dirname(__file__), "SN36291917.conf")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

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

# Named landmark indices
LM_NOSE = 0
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_ANKLE = 27
LM_RIGHT_ANKLE = 28


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
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), flow


# ── OSC Senders ──────────────────────────────────────────────

def send_pose_osc(client, results, prev_landmarks=None):
    """Send pose landmarks and derived control values via OSC."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks[0]

    # /zed/pose/landmarks — flat array of 33×4 (x, y, z, visibility)
    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z, lm.visibility])
    client.send_message("/zed/pose/landmarks", flat)

    # Key landmarks
    def send_lm(addr, idx):
        lm = landmarks[idx]
        client.send_message(addr, [lm.x, lm.y, lm.z])

    send_lm("/zed/pose/nose", LM_NOSE)
    send_lm("/zed/pose/left_wrist", LM_LEFT_WRIST)
    send_lm("/zed/pose/right_wrist", LM_RIGHT_WRIST)
    send_lm("/zed/pose/left_ankle", LM_LEFT_ANKLE)
    send_lm("/zed/pose/right_ankle", LM_RIGHT_ANKLE)

    # ── Derived control values ──
    lw = landmarks[LM_LEFT_WRIST]
    rw = landmarks[LM_RIGHT_WRIST]
    ls = landmarks[LM_LEFT_SHOULDER]
    rs = landmarks[LM_RIGHT_SHOULDER]
    lh = landmarks[LM_LEFT_HIP]
    rh = landmarks[LM_RIGHT_HIP]

    # Hand height: 0 = hip level, 1 = above head (clamped)
    # Use shoulder-hip distance as reference range
    torso_h = abs(ls.y - lh.y) + abs(rs.y - rh.y)
    torso_h = max(torso_h / 2, 0.01)

    hand_height_l = np.clip((lh.y - lw.y) / (torso_h * 2), 0.0, 1.0)
    hand_height_r = np.clip((rh.y - rw.y) / (torso_h * 2), 0.0, 1.0)
    client.send_message("/zed/control/hand_height_l", float(hand_height_l))
    client.send_message("/zed/control/hand_height_r", float(hand_height_r))

    # Hand distance: normalized by shoulder width
    shoulder_w = abs(rs.x - ls.x)
    shoulder_w = max(shoulder_w, 0.01)
    dx = rw.x - lw.x
    dy = rw.y - lw.y
    hand_dist = np.sqrt(dx**2 + dy**2) / (shoulder_w * 3)
    hand_dist = float(np.clip(hand_dist, 0.0, 1.0))
    client.send_message("/zed/control/hand_distance", hand_dist)

    # Body lean: left-right based on midpoint of shoulders vs midpoint of hips
    shoulder_mid_x = (ls.x + rs.x) / 2
    hip_mid_x = (lh.x + rh.x) / 2
    lean = (shoulder_mid_x - hip_mid_x) / max(shoulder_w, 0.01)
    lean = float(np.clip(lean, -1.0, 1.0))
    client.send_message("/zed/control/body_lean", lean)

    # Motion: average displacement from previous frame
    curr_arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    if prev_landmarks is not None:
        motion = float(np.mean(np.linalg.norm(curr_arr - prev_landmarks, axis=1)))
        motion = float(np.clip(motion * 10, 0.0, 1.0))  # scale up for sensitivity
    else:
        motion = 0.0
    client.send_message("/zed/control/motion", motion)

    return curr_arr


def send_depth_osc(client, depth_map):
    """Send depth summary via OSC."""
    # /zed/depth/average — global mean (normalized 0-1)
    avg = float(np.mean(depth_map) / 255.0)
    client.send_message("/zed/depth/average", avg)

    # /zed/depth/grid — 3×3 region averages (top-left → bottom-right)
    h, w = depth_map.shape[:2]
    grid = []
    for row in range(3):
        for col in range(3):
            y0 = row * h // 3
            y1 = (row + 1) * h // 3
            x0 = col * w // 3
            x1 = (col + 1) * w // 3
            region_avg = float(np.mean(depth_map[y0:y1, x0:x1]) / 255.0)
            grid.append(region_avg)
    client.send_message("/zed/depth/grid", grid)


def send_flow_osc(client, flow):
    """Send optical flow summary via OSC."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # /zed/flow/magnitude — overall motion (normalized 0-1)
    avg_mag = float(np.mean(mag))
    max_mag = float(np.max(mag)) if np.max(mag) > 0 else 1.0
    norm_mag = float(np.clip(avg_mag / max(max_mag, 1.0), 0.0, 1.0))
    client.send_message("/zed/flow/magnitude", norm_mag)

    # /zed/flow/direction — weighted average direction in degrees (0-360)
    weights = mag.flatten()
    total_weight = np.sum(weights)
    if total_weight > 0:
        # Use circular mean to handle wraparound
        angles_rad = ang.flatten()
        mean_sin = np.average(np.sin(angles_rad), weights=weights)
        mean_cos = np.average(np.cos(angles_rad), weights=weights)
        direction = float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)
    else:
        direction = 0.0
    client.send_message("/zed/flow/direction", direction)


# ── Main ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="ZED 2i Motion Tracker with OSC output")
    parser.add_argument("--osc-ip", default="127.0.0.1", help="OSC target IP (default: 127.0.0.1)")
    parser.add_argument("--osc-port", type=int, default=7500, help="OSC target port (default: 7500)")
    parser.add_argument("--no-osc", action="store_true", help="Disable OSC output")
    parser.add_argument("--no-syphon", action="store_true", help="Disable Syphon output")
    parser.add_argument("--no-vcam", action="store_true", help="Disable virtual camera output")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== ZED 2i Motion Tracker (macOS) ===\n")

    # ── OSC setup ──
    osc_client = None
    if not args.no_osc:
        osc_client = SimpleUDPClient(args.osc_ip, args.osc_port)
        print(f"OSC → {args.osc_ip}:{args.osc_port}")

    # ── Syphon setup ──
    syphon_server = None
    if not args.no_syphon and HAS_SYPHON:
        syphon_server = syphon.SyphonMetalServer("ZED Tracker")
        print("Syphon server: ZED Tracker")
    elif not args.no_syphon and not HAS_SYPHON:
        print("Syphon: not available (syphon-python not installed)")

    # ── Virtual camera setup ──
    vcam = None
    if not args.no_vcam and HAS_VCAM:
        try:
            vcam = pyvirtualcam.Camera(width=1280, height=720, fps=30, backend='obs')
            print(f"Virtual camera: {vcam.device} ({vcam.width}x{vcam.height})")
        except Exception as e:
            print(f"Virtual camera: failed ({e})")
    elif not args.no_vcam and not HAS_VCAM:
        print("Virtual camera: not available (pyvirtualcam not installed)")

    # Load calibration
    if not os.path.exists(CALIB_FILE):
        print(f"ERROR: Calibration file not found: {CALIB_FILE}")
        sys.exit(1)
    map_lx, map_ly, map_rx, map_ry, Q, focal, baseline = load_calibration(CALIB_FILE, RESOLUTION)

    # Open camera
    print(f"Opening camera index {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
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
    prev_landmarks = None
    frame_count = 0
    fps_timer = time.monotonic()
    measured_fps = 0.0

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

        # ── Always compute all data for OSC ──
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        # Pose (always)
        pose_display, pose_results = draw_pose(left_rect.copy(), landmarker, timestamp_ms)

        # Depth (always)
        depth_norm = compute_depth(left_rect, right_rect, left_matcher, right_matcher, wls_filter)

        # Optical flow (always)
        curr_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        flow_data = None
        if prev_gray is not None:
            flow_vis, flow_data = compute_optical_flow(prev_gray, curr_gray)
        else:
            flow_vis = left_rect.copy()
        prev_gray = curr_gray

        # ── Send OSC ──
        if osc_client is not None:
            client = osc_client
            # Pose
            prev_landmarks = send_pose_osc(client, pose_results, prev_landmarks)
            # Depth
            send_depth_osc(client, depth_norm)
            # Flow
            if flow_data is not None:
                send_flow_osc(client, flow_data)
            # System state
            client.send_message("/zed/mode", mode)
            client.send_message("/zed/fps", measured_fps)

        # ── Display (mode-dependent) ──
        if mode == 1:
            display = pose_display
        elif mode == 2:
            display = cv2.applyColorMap(depth_norm, depth_cmap)
        elif mode == 3:
            display = flow_vis
        elif mode == 4:
            depth_color = cv2.applyColorMap(depth_norm, depth_cmap)
            h = left_rect.shape[0]
            hw = left_rect.shape[1] // 2
            left_small = cv2.resize(left_rect, (hw, h // 2))
            depth_small = cv2.resize(depth_color, (hw, h // 2))
            display = np.hstack([left_small, depth_small])

        # HUD overlay
        label = f"{mode_names[mode]}"
        if mode in (2, 4):
            label += f" | {cmap_names[cmap_idx]} | disp={num_disp}"
        osc_label = f"OSC:{args.osc_port}" if osc_client else "OSC:off"
        syphon_label = "Sy:on" if syphon_server else "Sy:off"
        vcam_label = "VCam:on" if vcam else "VCam:off"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"FPS: {measured_fps:.0f} | {osc_label} | {syphon_label} | {vcam_label}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("ZED 2i Motion Tracker", display)

        # ── Syphon publish ──
        if syphon_server is not None:
            rgba = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA)
            sh, sw = rgba.shape[:2]
            texture = create_mtl_texture(syphon_server.device, sw, sh)
            copy_image_to_mtl_texture(rgba, texture)
            syphon_server.publish_frame_texture(texture, syphon.MtlTextureSize(sw, sh))

        # ── Virtual camera publish ──
        if vcam is not None:
            vcam_frame = display
            if vcam_frame.shape[1] != 1280 or vcam_frame.shape[0] != 720:
                vcam_frame = cv2.resize(vcam_frame, (1280, 720))
            vcam_frame = cv2.cvtColor(vcam_frame, cv2.COLOR_BGR2RGB)
            vcam.send(vcam_frame)

        # ── FPS measurement ──
        frame_count += 1
        elapsed = time.monotonic() - fps_timer
        if elapsed >= 1.0:
            measured_fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.monotonic()

        # ── Key handling ──
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

    # Cleanup
    if vcam is not None:
        vcam.close()
    if syphon_server is not None:
        syphon_server.stop()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
