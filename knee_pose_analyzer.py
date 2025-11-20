import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = 'yolo11n-pose.pt'   # Fast & small model
SAVE_FPS = 60.0
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
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (180 - angle)

# -----------------------------
# Step 1: Record with camera
# -----------------------------
def record_video():
    os.makedirs(SAVE_DIR_VID, exist_ok=True)
    model = YOLO(MODEL_PATH)

    # Use GPU (with half precision for speed)
    if torch.cuda.is_available():
        model.to("cuda")
        model.model.half()
        print("🚀 Using GPU in half precision mode")
    else:
        print("💻 Using CPU")

    cap = cv2.VideoCapture(0)
    recording = False
    out, filename = None, None

    print("Press 'r' to start/stop recording, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO pose inference
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()  # built-in visualization

        # Show live feed
        cv2.imshow("YOLO Pose Estimation", annotated_frame)
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

        if recording and out is not None:
            out.write(annotated_frame)

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
def analyze_video(video_path, analyze_side="both"):
    print(f"\n📊 Analyzing: {video_path}")
    model = YOLO(MODEL_PATH)
    os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)

    # Define keypoint indices
    left_leg = (11, 13, 15)
    right_leg = (12, 14, 16)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

        if keypoints is not None and len(keypoints) > 0:
            kp = keypoints[0]
            row = {"frame": frame_idx}

            if analyze_side in ["left", "both"]:
                left_angle = calculate_angle(kp[left_leg[0]], kp[left_leg[1]], kp[left_leg[2]])
                row["left_knee_angle"] = left_angle

            if analyze_side in ["right", "both"]:
                right_angle = calculate_angle(kp[right_leg[0]], kp[right_leg[1]], kp[right_leg[2]])
                row["right_knee_angle"] = right_angle

            data.append(row)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(data)
    if len(df) == 0:
        print("⚠️ No keypoints detected — check your video.")
        return

    # Smooth available angles
    for side in ["left", "right"]:
        if f"{side}_knee_angle" in df.columns:
            if len(df) > 11:
                df[f"{side}_knee_angle_smooth"] = savgol_filter(df[f"{side}_knee_angle"], 11, 3)
            else:
                df[f"{side}_knee_angle_smooth"] = df[f"{side}_knee_angle"]

    # --- Plot angles over time ---
    plt.figure(figsize=(8, 4))
    if "left_knee_angle_smooth" in df:
        plt.plot(df["frame"], df["left_knee_angle_smooth"], label="Left Knee", color="green")
    if "right_knee_angle_smooth" in df:
        plt.plot(df["frame"], df["right_knee_angle_smooth"], label="Right Knee", color="orange")
    plt.title("Knee Angle Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(SAVE_DIR_PLOTS, f"knee_angle_plot_{analyze_side}{timestamp}.png")
    plt.savefig(filename)
    plt.show()

    print(f"✅ Analysis complete! Saved plot: {filename}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Ask user for analysis mode
    print("\nChoose which leg(s) to analyze:")
    print("1 - Left leg")
    print("2 - Right leg")
    print("3 - Both legs")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        analyze_side = "left"
    elif choice == "2":
        analyze_side = "right"
    else:
        analyze_side = "both"

    video_file = record_video()
    if video_file:
        analyze_video(video_file, analyze_side)
