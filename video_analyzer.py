import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import torch

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = 'yolo11n-pose.pt'   # Fast & small model
SAVE_FPS = 30.0
SAVE_DIR_PLOTS = 'plots'

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
    return 2 * (180 - angle)  # straight = 0°, bent = increasing

# -----------------------------
# Extract one gait cycle
# -----------------------------
def extract_gait_cycle(angle_signal, fps=60):
    """Detect one gait cycle based on minima (heel strikes). Returns normalized 100-point cycle."""
    minima, _ = find_peaks(-angle_signal, distance=int(0.3 * fps))
    if len(minima) < 2:
        print("⚠️ Not enough gait cycles detected.")
        return None
    start, end = minima[0], minima[1]
    cycle = angle_signal[start:end]
    cycle_norm = np.interp(np.linspace(0, len(cycle) - 1, 100),
                           np.arange(len(cycle)), cycle)
    return cycle_norm

# -----------------------------
# Analyze video
# -----------------------------
def analyze_video(video_path, analyze_side="both"):
    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}")
        return

    print(f"\n📊 Analyzing: {video_path}")
    model = YOLO(MODEL_PATH)
    os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)

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
                row["left_knee_angle"] = calculate_angle(kp[left_leg[0]], kp[left_leg[1]], kp[left_leg[2]])

            if analyze_side in ["right", "both"]:
                row["right_knee_angle"] = calculate_angle(kp[right_leg[0]], kp[right_leg[1]], kp[right_leg[2]])

            data.append(row)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(data)
    if len(df) == 0:
        print("⚠️ No keypoints detected — check your video.")
        return

    # Smooth angles
    for side in ["left", "right"]:
        col = f"{side}_knee_angle"
        if col in df.columns:
            if len(df) > 11:
                df[f"{col}_smooth"] = savgol_filter(df[col], 11, 3)
            else:
                df[f"{col}_smooth"] = df[col]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # -------------------------------
    # Plot: Knee angle vs frame
    # -------------------------------
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
    filename = os.path.join(SAVE_DIR_PLOTS, f"knee_angle_plot_{analyze_side}_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    print(f"📈 Saved knee/frame plot: {filename}")

    # -------------------------------
    # Plot: Normalized gait cycle
    # -------------------------------
    for side in ["left", "right"]:
        if analyze_side not in [side, "both"]:
            continue
        col = f"{side}_knee_angle_smooth"
        if col not in df.columns:
            continue
        cycle = extract_gait_cycle(df[col].values, fps=SAVE_FPS)
        if cycle is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(np.linspace(0, 100, 100), cycle, label=f"{side.capitalize()} Knee")
            plt.xlabel("Gait Cycle (%)")
            plt.ylabel("Angle (degrees)")
            plt.title(f"{side.capitalize()} Knee - One Normalized Gait Cycle")
            plt.legend()
            plt.tight_layout()
            filename = os.path.join(SAVE_DIR_PLOTS, f"{side}_gait_cycle_{timestamp}.png")
            plt.savefig(filename)
            plt.show()
            print(f"📉 Saved gait cycle plot for {side}: {filename}")

    print("✅ Analysis complete!")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze knee angles in a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file to analyze")
    parser.add_argument("--side", type=str, default="both", choices=["left", "right", "both"],
                        help="Which leg(s) to analyze")
    args = parser.parse_args()

    analyze_video(args.video_path, analyze_side=args.side)
