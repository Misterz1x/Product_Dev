import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = 'yolo11n-pose.pt'
SAVE_DIR_PLOTS = 'plots'
SKIP_ROWS = 10  


# -----------------------------
# Helper: Calculate joint angle
# -----------------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return np.nan
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return 180 - angle  


# -----------------------------
# 1) Select and load MOT
# -----------------------------
def load_mot_columns(desired_columns):
    print("\n📄 Please select the .mot file that you want to analyze:")
    Tk().withdraw()
    filepath = askopenfilename(
        title="Select a .mot file",
        filetypes=[("MOT files", "*.mot"), ("All files", "*.*")]
    )
    if not filepath:
        raise ValueError("No file selected.")

    df = pd.read_csv(filepath, sep=r"\s+", skiprows=SKIP_ROWS, header=0)

    missing = [c for c in desired_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing MOT columns: {missing}")

    print("📄 Loaded MOT file:", filepath)
    return df[desired_columns], filepath


# -----------------------------
# 2) Select video file
# -----------------------------
def select_video_file():
    print("\n🎥 Please select the .mp4 (or other) video you want to analyze:")
    Tk().withdraw()
    filepath = askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not filepath:
        raise ValueError("No video file selected.")

    print("🎥 Selected video:", filepath)
    return filepath


# -----------------------------
# 3) Ask which side to analyze
# -----------------------------
def ask_side_selection():
    print("\n🔍 Please choose side to be analyzed: right, left or both")
    side = input("Type your choice: ").strip().lower()

    while side not in ["right", "left", "both"]:
        print("Invalid input. Please choose: right, left, both")
        side = input("Type your choice: ").strip().lower()

    print(f"➡️  Selected side for analysis: {side}")
    return side


# -----------------------------
# 4) Analyze video
# -----------------------------
def analyze_video(video_path, analyze_side="both"):

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)

    left_leg = (11, 13, 15)
    right_leg = (12, 14, 16)

    left_hip = (5, 11, 13)
    right_hip = (6, 12, 14)

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
            row = {
                "time_sec": frame_idx / fps
            }

            if analyze_side in ["left", "both"]:
                row["left_knee_angle"] = calculate_angle(kp[left_leg[0]], kp[left_leg[1]], kp[left_leg[2]])
                row["left_hip_angle"] = calculate_angle(kp[left_hip[0]], kp[left_hip[1]], kp[left_hip[2]])

            if analyze_side in ["right", "both"]:
                row["right_knee_angle"] = calculate_angle(kp[right_leg[0]], kp[right_leg[1]], kp[right_leg[2]])
                row["right_hip_angle"] = calculate_angle(kp[right_hip[0]], kp[right_hip[1]], kp[right_hip[2]])

            data.append(row)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ No keypoints detected.")
        return None

    # Smooth columns
    for side in ["left", "right"]:
        for joint in ["knee", "hip"]:
            col = f"{side}_{joint}_angle"
            if col in df.columns:
                df[f"{col}_smooth"] = (
                    savgol_filter(df[col], 11, 3) if len(df) > 11 else df[col]
                )

    print("✅ Video analysis complete.")
    return df


# -----------------------------
# 5) Plotting
# -----------------------------
def plot_angles(video_df, analyze_side):

    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Knee plot
    plt.figure(figsize=(8, 4))

    if analyze_side in ["left", "both"] and "left_knee_angle_smooth" in video_df:
        plt.plot(video_df["time_sec"], video_df["left_knee_angle_smooth"], label="Left Knee")

    if analyze_side in ["right", "both"] and "right_knee_angle_smooth" in video_df:
        plt.plot(video_df["time_sec"], video_df["right_knee_angle_smooth"], label="Right Knee")

    plt.title("Knee Angle Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Hip plot
    plt.figure(figsize=(8, 4))

    if analyze_side in ["left", "both"] and "left_hip_angle_smooth" in video_df:
        plt.plot(video_df["time_sec"], video_df["left_hip_angle_smooth"], label="Left Hip")

    if analyze_side in ["right", "both"] and "right_hip_angle_smooth" in video_df:
        plt.plot(video_df["time_sec"], video_df["right_hip_angle_smooth"], label="Right Hip")

    plt.title("Hip Angle Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# 6) Combine MOT + keypoint angles by aligning time
# -----------------------------
def merge_mot_and_video(mot_df, video_df):
    # Make sure MOT has a "time" column
    mot_times = mot_df["time"].values

    # Make sure video_df has a time column
    video_times = video_df["time_sec"].values

    # For each MOT time, find the nearest video time index
    nearest_idx = np.searchsorted(video_times, mot_times, side="left")

    # Fix indices that exceed bounds
    nearest_idx = np.clip(nearest_idx, 0, len(video_times)-1)

    # Create a new dataframe equal to the original MOT
    merged = mot_df.copy()

    # For all detected angle columns
    angle_cols = [c for c in video_df.columns if "angle" in c]

    # Add one column per angle, aligned to MOT time
    for col in angle_cols:
        merged[col] = video_df[col].iloc[nearest_idx].values

    print("🔗 Time-aligned MOT + video results merged.")
    return merged


# -----------------------------
# 7) Main workflow
# -----------------------------
def main():

    desired_columns = ["time", "hip_flexion_r", "knee_angle_r", "hip_flexion_l", "knee_angle_l"]

    # 1) Load MOT file
    mot_df, mot_path = load_mot_columns(desired_columns)

    # 2) Load video file
    video_path = select_video_file()

    # 3) Side selection
    analyze_side = ask_side_selection()

    # 4) Analyze video
    video_df = analyze_video(video_path, analyze_side)
    if video_df is None:
        print("❌ Video analysis failed.")
        return

    # 5) Plot
    plot_angles(video_df, analyze_side)

    # 6) Merge MOT + video summary
    combined = merge_mot_and_video(mot_df, video_df)

    # 7) Save result
    out_file = "csv_files/combined_results.csv"
    combined.to_csv(out_file, index=False)
    print("\n💾 Saved:", out_file)


# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    main()
