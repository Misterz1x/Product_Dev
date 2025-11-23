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
def extract_gait_cycle(angle_signal):
    # 1. Detect all minima (valleys in the signal)
    minima, _ = find_peaks(-angle_signal)

    if len(minima) < 2:
        print("⚠️ Not enough minima detected for gait cycle.")
        return None

    # 2. Compute distances between minima (step durations)
    diffs = np.diff(minima)

    # 3. Use the median step distance as the "true" gait cycle length
    cycle_len = int(np.median(diffs))

    if cycle_len < 5:
        print("⚠️ Detected cycle too short; skipping.")
        return None

    # 4. Use the first minimum as the start
    start = minima[0]
    end = start + cycle_len

    if end > len(angle_signal):
        print("⚠️ Not enough signal length for a full gait cycle.")
        return None

    cycle = angle_signal[start:end]

    # 5. Normalize to 100 points
    cycle_norm = np.interp(
        np.linspace(0, len(cycle)-1, 100),
        np.arange(len(cycle)),
        cycle
    )

    return cycle_norm


# -----------------------------
# Record video (real FPS)
# -----------------------------
def record_video():
    os.makedirs(SAVE_DIR_VID, exist_ok=True)
    model = YOLO(MODEL_PATH)

    # GPU setup
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

    print("Press 'r' to start/stop recording, 'q' to quit.")

    recording = False
    out, filename = None, None

    # ---------------------------
    # FPS Measurement Variables
    # ---------------------------
    prev_time = time.time()
    smoothed_fps = 0
    alpha = 0.1    # smoothing factor

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        # ---------------------------
        # FPS MEASUREMENT
        # ---------------------------
        now = time.time()
        instant_fps = 1 / (now - prev_time)
        prev_time = now

        # Smooth out the FPS so it doesn't jump
        if smoothed_fps == 0:
            smoothed_fps = instant_fps
        else:
            smoothed_fps = alpha * instant_fps + (1 - alpha) * smoothed_fps

        # Debug print (optional)
        # print(f"FPS: {smoothed_fps:.2f}")

        # ---------------------------
        # DISPLAY FRAME
        # ---------------------------
        cv2.imshow("YOLO Pose Estimation", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        # ---------------------------
        # START / STOP RECORDING
        # ---------------------------
        if key == ord('r'):
            if not recording:
                # Start recording
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(SAVE_DIR_VID, f"recording_{timestamp}.mp4")

                # Clamp FPS between 5 and 30 for stability
                fps_to_use = max(5, min(int(smoothed_fps), 30))

                h, w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps_to_use, (w, h))

                recording = True
                print(f"▶ Recording started at {fps_to_use} FPS → {filename}")

            else:
                # Stop recording
                recording = False
                out.release()
                print(f"⏹ Recording stopped: {filename}")

        # Save frames while recording
        if recording and out is not None:
            out.write(annotated_frame)

        # Quit
        if key == ord('q'):
            break

        # ---------------------------
        # NOTE:
        # Do *not* try to enforce FPS manually with sleep().
        # The FPS should be whatever your machine actually achieves.
        # ---------------------------

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    return filename



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

    left_hip = (5, 11, 13)
    right_hip = (6, 12, 14)

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

            if analyze_side in ["left", "both"]:
                row["left_hip_angle"] = calculate_angle(kp[left_hip[0]], kp[left_hip[1]], kp[left_hip[2]])

            if analyze_side in ["right", "both"]:
                row["right_hip_angle"] = calculate_angle(kp[right_hip[0]], kp[right_hip[1]], kp[right_hip[2]])

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

    for side in ["left", "right"]:
        col = f"{side}_hip_angle"
        if col in df.columns:
            if len(df) > 11:
                df[f"{col}_smooth"] = savgol_filter(df[col], 11, 3)
            else:
                df[f"{col}_smooth"] = df[col]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    

    # ----------------------------------------------------
    # Convert frames → seconds using the real video FPS
    # ----------------------------------------------------
    cap_check = cv2.VideoCapture(video_path)
    video_fps = cap_check.get(cv2.CAP_PROP_FPS)
    cap_check.release()

    df["time_sec"] = df["frame"] / video_fps
    print(f"🎞 Detected video FPS: {video_fps:.2f}")

    # -------------------------------
    # Plot: Knee angle vs frame
    # -------------------------------
    plt.figure(figsize=(8, 4))
    if "left_knee_angle_smooth" in df:
        plt.plot(df["time_sec"], df["left_knee_angle_smooth"], label="Left Knee", color="green")
    if "right_knee_angle_smooth" in df:
        plt.plot(df["time_sec"], df["right_knee_angle_smooth"], label="Right Knee", color="orange")
    plt.title("Knee Angle Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR_PLOTS, f"knee_angle_plot_{analyze_side}_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    print(f"📈 Saved knee/time plot: {filename}")


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
        cycle = extract_gait_cycle(df[col].values)

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


    # -------------------------------
    # Plot: Hip angle vs frame
    # -------------------------------
    plt.figure(figsize=(8, 4))
    if "left_hip_angle_smooth" in df:
        plt.plot(df["time_sec"], df["left_hip_angle_smooth"], label="Left Hip", color="blue")
    if "right_hip_angle_smooth" in df:
        plt.plot(df["time_sec"], df["right_hip_angle_smooth"], label="Right Hip", color="red")
    plt.title("Hip Angle Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR_PLOTS, f"hip_angle_plot_{analyze_side}_{timestamp}.png")
    plt.savefig(filename)
    plt.show()
    print(f"📈 Saved hip/time plot: {filename}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\nChoose which leg(s) to analyze:")
    print("1 - Left leg")
    print("2 - Right leg")
    print("3 - Both legs")

    choice = input("Enter choice (1/2/3): ").strip()
    analyze_side = "both" if choice not in ["1", "2"] else ("left" if choice == "1" else "right")

    # Record the video
    video_file = record_video()

    # If recording was successful
    if video_file:
        analyze_video(video_file, analyze_side)

