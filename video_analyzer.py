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
from matplotlib.widgets import Cursor


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
# 5) Combine MOT + keypoint angles by aligning time
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
# 6) Select gait cycle
# -----------------------------

def select_gait_cycle(df, analyze_side):
    """
    Shows synchronized plots (video + MOT)
    and allows selecting start/end time.
    Only displays the selected side.
    Adds a visible line after first click.
    """

    # Determine columns based on side
    video_cols = {
        "left":  ["left_knee_angle_smooth", "left_hip_angle_smooth"],
        "right": ["right_knee_angle_smooth", "right_hip_angle_smooth"],
        "both":  [
            "left_knee_angle_smooth", "right_knee_angle_smooth",
            "left_hip_angle_smooth",  "right_hip_angle_smooth"
        ]
    }

    mot_cols = {
        "left":  ["knee_angle_l", "hip_flexion_l"],
        "right": ["knee_angle_r", "hip_flexion_r"],
        "both":  [
            "knee_angle_l", "knee_angle_r",
            "hip_flexion_l","hip_flexion_r"
        ]
    }

    # Create 4 plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    time = df["time"]

    # 1 - Knee Video
    axs[0].set_title("Knee Angle (Video)")
    for col in video_cols[analyze_side]:
        if "knee" in col and col in df.columns:
            axs[0].plot(time, df[col], label=col)
    axs[0].legend()

    # 2 - Hip Video
    axs[1].set_title("Hip Angle (Video)")
    for col in video_cols[analyze_side]:
        if "hip" in col and col in df.columns:
            axs[1].plot(time, df[col], label=col)
    axs[1].legend()

    # 3 - Knee MOT
    axs[2].set_title("Knee Angle (MOT)")
    for col in mot_cols[analyze_side]:
        if "knee" in col and col in df.columns:
            axs[2].plot(time, df[col], label=col)
    axs[2].legend()

    # 4 - Hip MOT
    axs[3].set_title("Hip Angle (MOT)")
    for col in mot_cols[analyze_side]:
        if "hip" in col and col in df.columns:
            axs[3].plot(time, df[col], label=col)
    axs[3].legend()

    axs[-1].set_xlabel("Time (seconds)")

    # Cursor
    cursor = Cursor(axs[0], useblit=True, color='red', linewidth=1)

    click_times = []
    point_line = None

    def onclick(event):
        nonlocal point_line

        if event.inaxes not in axs:
            return

        click_times.append(event.xdata)
        print(f"Selected time: {event.xdata:.3f} sec")

        # After first click → draw vertical line
        if len(click_times) == 1:
            for ax in axs:
                point_line = ax.axvline(event.xdata, color='blue', linestyle='--')
            fig.canvas.draw()

        # After second click → close figure
        if len(click_times) == 2:
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)

    print("\n👉 Click ONCE for gait cycle START, ONCE for gait cycle END.")
    plt.ylabel("Angle (degrees)")
    plt.show()

    # Must have 2 points
    if len(click_times) < 2:
        print("❌ Not enough points selected.")
        return None

    # Determine cycle window
    t_start = min(click_times)
    t_end = max(click_times)

    df_cycle = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()

    if df_cycle.empty:
        print("❌ Selected window contains no data.")
        return None

    # Normalize to 0–100%
    t0, t1 = df_cycle["time"].iloc[0], df_cycle["time"].iloc[-1]
    df_cycle["time_norm"] = (df_cycle["time"] - t0) / (t1 - t0) * 100

    print(f"\n✅ Gait cycle extracted from {t0:.3f} to {t1:.3f} seconds.")
    return df_cycle


# -----------------------------
# 7) Plotting normalized gait cycle
# -----------------------------
def plot_normalized_gait_cycle(df_gait, analyze_side):
    """
    Plots the selected gait cycle from 0–100%, 
    showing only right, left, or both sides.

    analyze_side: "right", "left", or "both"
    """

    # Close any previous figures
    plt.close("all")

    # Columns and subplot titles
    columns = ["right_knee_angle_smooth", "right_hip_angle_smooth",
               "knee_angle_r", "hip_flexion_r"]
    titles = ["Right Knee Angle (Video)", "Right Hip Angle (Video)",
              "Right Knee Angle (MOT)", "Right Hip Angle (MOT)"]

    # Filter out missing columns
    columns = [c for c in columns if c in df_gait.columns]
    titles = titles[:len(columns)]

    if not columns:
        print("❌ None of the required columns are in the dataframe.")
        return

    gait_cycle = df_gait["time_norm"]

    # Create figure
    fig, axs = plt.subplots(len(columns), 1, figsize=(10, 3 * len(columns)), sharex=True)

    if len(columns) == 1:
        axs = [axs]

    # Plot each column
    for ax, col, title in zip(axs, columns, titles):
        ax.plot(gait_cycle, df_gait[col], label=col)
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Gait Cycle (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_PLOTS, f"gait_cycle_{analyze_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    plt.show()

# -----------------------------
# 8) Main workflow
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

    # 5) Merge MOT + video summary
    combined = merge_mot_and_video(mot_df, video_df)

    # 6) Select gait cycle
    gait_cycle_df = select_gait_cycle(combined, analyze_side)

    if gait_cycle_df is not None:
        out_file_cycle = "csv_files/gait_cycle_results.csv"
        gait_cycle_df.to_csv(out_file_cycle, index=False)
        print("💾 Saved gait cycle data:", out_file_cycle)

    # 7) Save result
    out_file = "csv_files/combined_results.csv"
    combined.to_csv(out_file, index=False)
    print("\n💾 Saved:", out_file)

    # 8) Plot normalized gait cycle

    if gait_cycle_df is not None:
        plot_normalized_gait_cycle(gait_cycle_df, analyze_side)


# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    main()
