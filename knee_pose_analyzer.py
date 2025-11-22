import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import torch
import time

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = 'yolo11n-pose.pt'   # Fast & small model
SAVE_DIR_VID = 'data_recordings'
SAVE_DIR_PLOTS = 'data_plots'

# -----------------------------
# Helper: Calculate joint angle
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b

    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    if mag1 == 0 or mag2 == 0:
        raise ValueError("Zero-length vector encountered in angle calculation.")

    cosine = np.dot(v1, v2) / (mag1 * mag2)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (180 - angle)

# -----------------------------
# Extract one gait cycle
# -----------------------------
def extract_gait_cycle(angle_signal, fps=30):
    minima, _ = find_peaks(-angle_signal, distance=int(0.3 * fps))
    if len(minima) < 2:
        print("⚠️ Not enough gait cycles detected.")
        return None
    start, end = minima[0], minima[1]
    cycle = angle_signal[start:end]
    cycle_norm = np.interp(np.linspace(0, len(cycle)-1, 100), np.arange(len(cycle)), cycle)
    return cycle_norm

# -----------------------------
# Record video (real FPS)
# -----------------------------
def record_video():
    os.makedirs(SAVE_DIR_VID, exist_ok=True)
    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available():
        model.to("cuda")
        model.model.half()
        print("🚀 Using GPU in half precision mode")
    else:
        print("💻 Using CPU")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return None

    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera FPS detected: {camera_fps}")
    frame_interval = 1.0 / camera_fps

    recording = False
    out, filename = None, None
    print("Press 'r' to start/stop recording, 'q' to quit.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Pose Estimation", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            if not recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(SAVE_DIR_VID, f"recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, camera_fps,
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

        # Ensure real-time FPS
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    return filename


# -----------------------------
# Analyze video
# -----------------------------
def analyze_video(video_path, analyze_side="both", video_fps=30.0):
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

        # <-- Pass actual video FPS here -->
        cycle = extract_gait_cycle(df[col].values, fps=video_fps)

        if cycle is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(np.linspace(0, 100, 100), cycle, label=f"{side.capitalize()} Knee")
            plt.xlabel("% Gait Cycle")
            plt.ylabel("Angle (degrees)")
            plt.title(f"{side.capitalize()} Knee - One Normalized Gait Cycle")
            plt.legend()
            plt.tight_layout()

            filename_cycle = os.path.join(SAVE_DIR_PLOTS, f"{side}_gait_cycle_{timestamp}.png")
            plt.savefig(filename_cycle)
            plt.show()
            print(f"📉 Saved gait cycle plot for {side}: {filename_cycle}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\nChoose which leg(s) to analyze:")
    print("1 - Left leg")
    print("2 - Right leg")
    print("3 - Both legs")
    choice = input("Enter choice (1/2/3): ").strip()
    analyze_side = "both" if choice not in ["1", "2"] else ("left" if choice=="1" else "right")

    video_file = record_video()
    if video_file:
        cap = cv2.VideoCapture(video_file)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        analyze_video(video_file, analyze_side, video_fps=actual_fps)

