import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate, stats
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
# Helper: Calculate knee angle
# -----------------------------
def calculate_knee_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return np.nan
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return 180 - angle  


# -----------------------------
# Helper: Calculate hip angle
# -----------------------------
def calculate_hip_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    v1 = a - b   # proximal (trunk)
    v2 = c - b   # distal (thigh)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return np.nan

    # Original magnitude calculation (UNCHANGED)
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle = 180 - angle   # keep your original method

    # Add SIGN using 2D cross product
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    sign = np.sign(cross)

    return angle * sign

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
                row["left_knee_angle"] = calculate_knee_angle(kp[left_leg[0]], kp[left_leg[1]], kp[left_leg[2]])
                row["left_hip_angle"] = calculate_hip_angle(kp[left_hip[0]], kp[left_hip[1]], kp[left_hip[2]])

            if analyze_side in ["right", "both"]:
                row["right_knee_angle"] = calculate_knee_angle(kp[right_leg[0]], kp[right_leg[1]], kp[right_leg[2]])
                row["right_hip_angle"] = calculate_hip_angle(kp[right_hip[0]], kp[right_hip[1]], kp[right_hip[2]])

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
def select_gait_cycles(df, analyze_side):
    """
    Shows synchronized plots (video + MOT) and allows selecting 
    MULTIPLE gait cycles.
    
    - Click START then END for Cycle 1.
    - Click START then END for Cycle 2, etc.
    - Close the window when finished.
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

    # --- Plotting Helper ---
    def plot_channels(ax, title, col_list):
        ax.set_title(title)
        for col in col_list:
            if col in df.columns:
                ax.plot(time, df[col], label=col)
        ax.legend(loc='upper right', fontsize='small')

    # 1 - Knee Video
    plot_channels(axs[0], "Knee Angle (Video)", [c for c in video_cols[analyze_side] if "knee" in c])
    # 2 - Hip Video
    plot_channels(axs[1], "Hip Angle (Video)", [c for c in video_cols[analyze_side] if "hip" in c])
    # 3 - Knee MOT
    plot_channels(axs[2], "Knee Angle (MOT)", [c for c in mot_cols[analyze_side] if "knee" in c])
    # 4 - Hip MOT
    plot_channels(axs[3], "Hip Angle (MOT)", [c for c in mot_cols[analyze_side] if "hip" in c])

    axs[-1].set_xlabel("Time (seconds)")

    # Cursor
    cursor = Cursor(axs[0], useblit=True, color='red', linewidth=1)

    click_times = []

    def onclick(event):
        if event.inaxes not in axs:
            return

        # Store time
        t_click = event.xdata
        click_times.append(t_click)
        click_count = len(click_times)
        
        print(f"Click {click_count}: {t_click:.3f} sec")

        # Visual Feedback
        # 1. Draw vertical line for every click
        for ax in axs:
            ax.axvline(t_click, color='blue', linestyle='--', alpha=0.6)

        # 2. If we just finished a pair (Even number), shade the region
        if click_count % 2 == 0:
            t_start = click_times[-2]
            t_end = click_times[-1]
            # Ensure correct order for shading if user clicked backwards
            span_min, span_max = min(t_start, t_end), max(t_start, t_end)
            
            for ax in axs:
                ax.axvspan(span_min, span_max, color='green', alpha=0.15)
            print(f"   -> Cycle pair recorded ({span_min:.2f} to {span_max:.2f})")

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    print("\n" + "="*60)
    print("👉 INSTRUCTIONS:")
    print("1. Click START, then Click END for a gait cycle.")
    print("2. Repeat for as many cycles as you want.")
    print("3. Close the figure window to finish processing.")
    print("="*60 + "\n")
    
    plt.ylabel("Angle (degrees)")
    plt.show() # Code pauses here until window is closed

    # --- Post-Processing ---

    # 1. Ensure even number of clicks
    if len(click_times) % 2 != 0:
        print(f"\n⚠️ Warning: Odd number of clicks ({len(click_times)}). Removing last click to enforce Start/End pairs.")
        click_times.pop()

    # 2. Check if we have enough data
    if len(click_times) < 2:
        print("❌ No complete gait cycles selected.")
        return None

    # 3. Extract and combine cycles
    all_cycles = []
    
    # Iterate in steps of 2: (0,1), (2,3), (4,5)...
    for i in range(0, len(click_times), 2):
        t1 = click_times[i]
        t2 = click_times[i+1]
        
        # Sort in case user clicked End before Start
        t_start, t_end = min(t1, t2), max(t1, t2)
        
        # Slice DataFrame
        df_cycle = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()
        
        if df_cycle.empty:
            print(f"⚠️ Warning: Cycle {i//2 + 1} empty (no data points). Skipping.")
            continue

        # Normalize Time (0-100%)
        t0_local = df_cycle["time"].iloc[0]
        duration = df_cycle["time"].iloc[-1] - t0_local
        
        # Avoid division by zero
        if duration == 0:
            df_cycle["time_norm"] = 0
        else:
            df_cycle["time_norm"] = (df_cycle["time"] - t0_local) / duration * 100

        # Add Cycle ID (1-based index)
        df_cycle["cycle_id"] = (i // 2) + 1
        
        all_cycles.append(df_cycle)

    if not all_cycles:
        return None

    # Combine into one big DataFrame
    final_df = pd.concat(all_cycles, ignore_index=True)
    
    print(f"\n✅ Successfully extracted {len(all_cycles)} gait cycles.")
    return final_df


# -----------------------------
# 7) Plotting normalized gait cycle
# -----------------------------
def plot_normalized_gait_cycles(df_gait, analyze_side):
    """
    Plots multiple gait cycles from 0–100%.
    - Resamples all cycles to a common 101-point grid.
    - Plots individual cycles (thin/faint).
    - Plots the MEAN of those cycles (thick/solid).
    - Saves to global SAVE_DIR_PLOTS.
    """
    
    # 1. Setup Standard Grid (0 to 100% with 101 points)
    common_x = np.linspace(0, 100, 101)
    cycle_ids = df_gait['cycle_id'].unique()
    
    print(f"Processing {len(cycle_ids)} cycles for '{analyze_side}' side...")

    # 2. Define Variables
    variables = [
        ("knee_angle_video", "right_knee_angle_smooth", "left_knee_angle_smooth", "Knee Angle (Video)"),
        ("hip_angle_video", "right_hip_angle_smooth", "left_hip_angle_smooth", "Hip Angle (Video)"),
        ("knee_angle_mot", "knee_angle_r", "knee_angle_l", "Knee Angle (MOT)"),
        ("hip_angle_mot", "hip_flexion_r", "hip_flexion_l", "Hip Angle (MOT)")
    ]

    # 3. Build Plot List based on side 
    plot_items = []
    
    # We iterate over the 4 required plots
    for i, (var_name, col_r, col_l, title) in enumerate(variables):
        cols_to_plot = []
        
        # Determine the angle type (e.g., 'Knee' or 'Hip') based on the title
        # This is the new, robust check to ensure we aren't mixing data.
        required_angle_type = 'Knee' if 'Knee' in title else 'Hip'

        # Helper function to check column relevance
        def is_relevant(col, required_type):
            # Check if the column exists AND contains the required angle type (e.g., 'knee' or 'hip')
            # The .lower() prevents errors if titles/columns use different cases.
            return col in df_gait.columns and required_type.lower() in col.lower()

        # --- Filtering Logic ---
        
        if analyze_side == "right":
            if is_relevant(col_r, required_angle_type):
                cols_to_plot.append((col_r, 'tab:blue', 'Right'))
            
        elif analyze_side == "left":
            if is_relevant(col_l, required_angle_type):
                cols_to_plot.append((col_l, 'tab:orange', 'Left'))

        elif analyze_side == "both":
            if is_relevant(col_r, required_angle_type):
                cols_to_plot.append((col_r, 'tab:blue', 'Right'))
            if is_relevant(col_l, required_angle_type):
                cols_to_plot.append((col_l, 'tab:orange', 'Left'))
            
        # If any columns were found for this variable, add the plot item
        if cols_to_plot:
            plot_items.append((cols_to_plot, title))

    if not plot_items:
        print("❌ No valid columns found for plotting.")
        return

    # 4. Create Subplots
    fig, axs = plt.subplots(len(plot_items), 1, figsize=(10, 3.5 * len(plot_items)), sharex=True)
    if len(plot_items) == 1: axs = [axs]

    # 5. Processing and Plotting Loop 
    for ax, (cols_info, title) in zip(axs, plot_items):
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Angle (°)")
        
        for col_name, color, label_prefix in cols_info:
            # Container for resampled arrays
            resampled_cycles = []

            # --- A. Interpolate and Plot Individual Cycles ---
            for cid in df_gait['cycle_id'].unique():
                subset = df_gait[df_gait['cycle_id'] == cid]
                
                if len(subset) < 2:
                    # print(f"Skipping cycle {cid} for {col_name}: Insufficient data points.")
                    continue

                f = interpolate.interp1d(subset['time_norm'], subset[col_name], 
                                         kind='linear', fill_value="extrapolate")
                
                y_new = f(common_x)
                resampled_cycles.append(y_new)
                
                # Plot individual cycle (Thin, transparent)
                ax.plot(common_x, y_new, color=color, alpha=0.25, linewidth=1)
            
            # --- B. Calculate and Plot Mean ---
            if not resampled_cycles:
                continue
                
            data_matrix = np.array(resampled_cycles)
            mean_curve = np.mean(data_matrix, axis=0)
            
            # Plot Mean (Thick, Solid)
            ax.plot(common_x, mean_curve, color=color, linewidth=2.5, label=f"{label_prefix} Mean")
            
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize='small')

    axs[-1].set_xlabel("Gait Cycle (%)")
    plt.tight_layout()

    # 6. Save using global SAVE_DIR_PLOTS
    if 'SAVE_DIR_PLOTS' in globals():
        # Ensure directory exists
        os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)
        
        filename = f"gait_analysis_{analyze_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(SAVE_DIR_PLOTS, filename)
        
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to: {save_path}")
    else:
        print("⚠️ SAVE_DIR_PLOTS not defined. Plot not saved.")

    plt.show()



# -----------------------------
# 8) Compare methods
# -----------------------------
def ICC2_1(data):
    """
    Computes Interclass Correlation Coefficient ICC(2,1) — two-way random, absolute agreement, single rater.
    data should be Nxk (subjects x methods).
    """
    data = np.asarray(data)
    n, k = data.shape

    mean_raters = np.mean(data, axis=0)
    mean_subjects = np.mean(data, axis=1)
    grand_mean = np.mean(data)

    # Mean squares
    MS_subjects = np.sum((mean_subjects - grand_mean)**2) * k / (n - 1)
    MS_raters   = np.sum((mean_raters - grand_mean)**2) * n / (k - 1)
    MS_error    = np.sum((data - mean_subjects[:, None] - mean_raters + grand_mean)**2) / ((k - 1)*(n - 1))

    ICC = (MS_subjects - MS_error) / (MS_subjects + (k - 1)*MS_error + k*(MS_raters - MS_error)/n)
    return ICC


def compare_gait_methods(df_gait, analyze_side):
    """
    Compares cycle-mean VIDEO vs MOT measurement methods for:
        - Knee angles
        - Hip angles

    Returns metrics:
        - Absolute Error (array)
        - MAE
        - Mean Bias
        - ICC
        - Cross-correlation coefficient
        - Pearson correlation coefficient
    """

    # Common normalized grid
    common_x = np.linspace(0, 100, 101)
    cycle_ids = df_gait['cycle_id'].unique()

    # ============================
    # Variables to compare
    # ============================
    variable_pairs = [
        ("knee_angle_video", "right_knee_angle_smooth", "left_knee_angle_smooth",
         "knee_angle_mot",   "knee_angle_r",           "knee_angle_l",
         "Knee Angle"),
        
        ("hip_angle_video",  "right_hip_angle_smooth", "left_hip_angle_smooth",
         "hip_angle_mot",    "hip_flexion_r",          "hip_flexion_l",
         "Hip Angle")
    ]

    results = {}

    for video_name, video_r, video_l, mot_name, mot_r, mot_l, label in variable_pairs:

        # Pick appropriate side columns
        def pick(col_r, col_l):
            if analyze_side == "right": return col_r
            if analyze_side == "left":  return col_l
            if analyze_side == "both":  # average both later
                return (col_r, col_l)
        
        col_video = pick(video_r, video_l)
        col_mot   = pick(mot_r,   mot_l)

        # Skip if columns absent
        def col_exists(c):
            if isinstance(c, tuple):
                return all([(cc in df_gait.columns) for cc in c])
            return c in df_gait.columns

        if not col_exists(col_video) or not col_exists(col_mot):
            print(f"Skipping {label}: Missing required columns.")
            continue

        # -----------------------------------
        # A) Extract and resample cycles
        # -----------------------------------
        video_cycles = []
        mot_cycles   = []

        for cid in cycle_ids:
            subset = df_gait[df_gait['cycle_id'] == cid]
            if len(subset) < 2:
                continue

            # helper
            def interp_col(col):
                if isinstance(col, tuple):
                    # both sides → average of interpolated curves
                    f1 = interpolate.interp1d(subset['time_norm'], subset[col[0]],
                                              kind="linear", fill_value="extrapolate")
                    f2 = interpolate.interp1d(subset['time_norm'], subset[col[1]],
                                              kind="linear", fill_value="extrapolate")
                    return (f1(common_x) + f2(common_x)) / 2
                else:
                    f = interpolate.interp1d(subset['time_norm'], subset[col],
                                             kind="linear", fill_value="extrapolate")
                    return f(common_x)

            video_cycles.append(interp_col(col_video))
            mot_cycles.append(interp_col(col_mot))

        if len(video_cycles) == 0:
            print(f"No valid cycles for {label}.")
            continue

        video_cycles = np.array(video_cycles)
        mot_cycles   = np.array(mot_cycles)

        # -----------------------------------
        # B) Compute mean curves
        # -----------------------------------
        mean_video = np.mean(video_cycles, axis=0)
        mean_mot   = np.mean(mot_cycles,   axis=0)

        # -----------------------------------
        # C) Compute metrics
        # -----------------------------------
        diff = mean_video - mean_mot

        abs_error = np.abs(diff)
        mae = np.mean(abs_error)
        mean_bias = np.mean(diff)

        # ICC (2,1)
        icc = ICC2_1(np.vstack([mean_video, mean_mot]).T)

        # Cross correlation
        cross_corr = np.max(np.correlate(mean_video - mean_video.mean(),
                                         mean_mot   - mean_mot.mean(),
                                         mode="full"))
        cross_corr /= (np.std(mean_video) * np.std(mean_mot) * len(mean_video))

        # Pearson r
        pearson_r, _ = stats.pearsonr(mean_video, mean_mot)

        results[label] = {
            "mean_video": mean_video,
            "mean_mot":   mean_mot,
            "abs_error_curve": abs_error,
            "MAE": mae,
            "mean_bias": mean_bias,
            "ICC2_1": icc,
            "cross_correlation": cross_corr,
            "pearson_r": pearson_r
        }

    return results

# -----------------------------
# 9) Main workflow
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
    gait_cycle_df = select_gait_cycles(combined, analyze_side)

    if gait_cycle_df is not None:
        out_file_cycle = "csv_files/gait_cycle_results.csv"

        # Ensure the folder exists
        os.makedirs(os.path.dirname(out_file_cycle), exist_ok=True)

        # Save gait cycle data
        gait_cycle_df.to_csv(out_file_cycle, index=False)
        print("💾 Saved gait cycle data:", out_file_cycle)

    # 7) Save result
    out_file = "csv_files/combined_results.csv"
    combined.to_csv(out_file, index=False)
    print("\n💾 Saved:", out_file)

    # 8) Plot normalized gait cycle

    if gait_cycle_df is not None:
        plot_normalized_gait_cycles(gait_cycle_df, analyze_side)

    # 9) Compare methods
    if gait_cycle_df is not None:
        comparison_results = compare_gait_methods(gait_cycle_df, analyze_side)
        for joint, metrics in comparison_results.items():
            print(f"\n📊 {joint} Comparison Metrics:")
            print(f" - MAE: {metrics['MAE']:.2f} degrees")
            print(f" - Mean Bias: {metrics['mean_bias']:.2f} degrees")
            print(f" - ICC(2,1): {metrics['ICC2_1']:.3f}")
            print(f" - Cross-correlation: {metrics['cross_correlation']:.3f}")
            print(f" - Pearson r: {metrics['pearson_r']:.3f}")


# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    main()
