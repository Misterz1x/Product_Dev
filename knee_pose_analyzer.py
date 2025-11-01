import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = 'yolov8n-pose.pt'
SAVE_FPS = 30.0
KNEE_SIDE = 'left'  # choose 'left' or 'right'
SAVE_DIR_VID = 'data_recordings'
SAVE_DIR_PLOTS = 'data_plots'


# -----------------------------
# Helper: Calculate joint angle
# -----------------------------
def calculate_angle(a, b, c):
    """Calculate the angle at point b given three points (a-b-c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# -----------------------------
# Helper: Draw selected keypoints
# -----------------------------
def draw_selected_leg(frame, keypoints, side='left'):
    """Draw only the hip_knee_ankle keypoints for the chosen leg."""
    if side == 'left':
        hip_idx, knee_idx, ankle_idx = 11, 13, 15
        color = (0, 255, 0)
    else:
        hip_idx, knee_idx, ankle_idx = 12, 14, 16
        color = (0, 128, 255)

    pts = [keypoints[hip_idx], keypoints[knee_idx], keypoints[ankle_idx]]

    # Draw small circles and connecting lines
    for pt in pts:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, color, -1)
    cv2.line(frame, (int(pts[0][0]), int(pts[0][1])),
             (int(pts[1][0]), int(pts[1][1])), color, 3)
    cv2.line(frame, (int(pts[1][0]), int(pts[1][1])),
             (int(pts[2][0]), int(pts[2][1])), color, 3)
    return frame


# -----------------------------
# Step 1: Record with camera
# -----------------------------
def record_video():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    # Create save directory if missing
    os.makedirs(SAVE_DIR_VID, exist_ok=True)

    recording = False
    out = None
    filename = None

    print("Press 'r' to start/stop recording, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO pose inference
        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

        annotated_frame = frame.copy()
        if keypoints is not None and len(keypoints) > 0:
            # Use the first detected person
            annotated_frame = draw_selected_leg(annotated_frame, keypoints[0], side=KNEE_SIDE)

        # Display live view
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        # Toggle recording
        if key == ord('r'):
            if not recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(SAVE_DIR_VID, f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, SAVE_FPS,
                                      (frame.shape[1], frame.shape[0]))
                recording = True
                print(f"▶ Recording started: {filename}")
            else:
                recording = False
                out.release()
                print(f"⏹ Recording stopped: {filename}")

        # Save annotated frames if recording
        if recording and out is not None:
            out.write(annotated_frame)

        # Quit
        if key == ord('q'):
            break

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

    return filename


# -----------------------------
# Step 2: Analyze the video
# -----------------------------
def analyze_video(video_path):
    print(f"\n📊 Analyzing: {video_path}")
    model = YOLO(MODEL_PATH)

    # Create save directory if missing
    os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)

    if KNEE_SIDE == 'left':
        hip_idx, knee_idx, ankle_idx = 11, 13, 15
    else:
        hip_idx, knee_idx, ankle_idx = 12, 14, 16

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

        if keypoints is not None and len(keypoints) > 0:
            kp = keypoints[0]
            hip, knee, ankle = kp[hip_idx], kp[knee_idx], kp[ankle_idx]
            angle = calculate_angle(hip, knee, ankle)
            data.append({
                "frame": frame_idx,
                "knee_angle": angle,
                "knee_x": knee[0],
                "knee_y": knee[1]
            })

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(data)
    if len(df) == 0:
        print("⚠️ No keypoints detected — check your video.")
        return

    # Smooth angle values
    if len(df) > 11:
        df["knee_angle_smooth"] = savgol_filter(df["knee_angle"], 11, 3)
    else:
        df["knee_angle_smooth"] = df["knee_angle"]

    # --- Plot angle over time ---
    plt.figure(figsize=(8, 4))
    plt.plot(df["frame"], df["knee_angle"], label="Raw", alpha=0.5)
    plt.plot(df["frame"], df["knee_angle_smooth"], label="Smoothed", linewidth=2)
    plt.title(f"{KNEE_SIDE.capitalize()} Knee Angle Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_PLOTS, f"{KNEE_SIDE}_knee_angle_plot.png"))
    plt.show()

    # --- Plot trajectory ---
    plt.figure(figsize=(4, 6))
    plt.plot(df["knee_x"], df["knee_y"], marker="o", markersize=2)
    plt.gca().invert_yaxis()
    plt.title(f"{KNEE_SIDE.capitalize()} Knee Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_PLOTS, f"{KNEE_SIDE}_knee_trajectory_plot.png"))
    plt.show()

    print("✅ Analysis complete!")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    video_file = record_video()
    if video_file:
        analyze_video(video_file)
