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
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

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
def ICC2_1_matrix(data):
    """
    Compute ICC(2,1) on a data matrix shaped (subjects, raters).
    Two-way random, absolute agreement, single rater.
    Returns ICC float.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("ICC2_1_matrix expects a 2D array (subjects x raters).")
    n, k = data.shape  # n subjects, k raters

    # means
    mean_subject = np.mean(data, axis=1, keepdims=True)  # n x 1
    mean_rater = np.mean(data, axis=0, keepdims=True)    # 1 x k
    grand_mean = np.mean(data)

    # Sum squares
    SS_subjects = np.sum((mean_subject - grand_mean)**2) * k
    SS_raters = np.sum((mean_rater - grand_mean)**2) * n
    SS_total = np.sum((data - grand_mean)**2)
    SS_error = SS_total - SS_subjects - SS_raters

    MS_subjects = SS_subjects / (n - 1)
    MS_raters = SS_raters / (k - 1)
    MS_error = SS_error / ((k - 1) * (n - 1))

    # ICC(2,1)
    denom = MS_subjects + (k - 1) * MS_error + (k * (MS_raters - MS_error) / n)
    if denom == 0:
        return np.nan
    ICC = (MS_subjects - MS_error) / denom
    return ICC

def max_normalized_crosscorr(a, b):
    """
    Compute maximum normalized cross-correlation between 1D arrays a and b.
    Returns float in [-1,1].
    """
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    corr = np.correlate(a, b, mode='full')
    denom = np.std(a) * np.std(b) * len(a)
    if denom == 0:
        return np.nan
    return np.max(corr) / denom

# ---------------------------
# Core comparison function
# ---------------------------
def compare_gait_methods(df_gait, analyze_side="right"):
    """
    Extract cycles, resample to 0-100% (101 points), compute mean curves,
    compute per-cycle MAE and bias, and overall metrics.

    Returns a results dict with entries for 'Knee Angle' and 'Hip Angle'.
    Each entry contains:
      - mean_video (101,)
      - mean_mot   (101,)
      - sd_video (101,)
      - sd_mot   (101,)
      - abs_error_curve (101,)
      - MAE_curve (float)
      - mean_bias (float)
      - pearson_r (float)
      - cross_corr_max (float)
      - ICC_time (ICC computed with 101 subjects x 2 raters using mean curves)
      - ICC_cycles (ICC computed across cycles using per-cycle mean values)
      - per_cycle_MAE (array length = n_cycles)
      - per_cycle_bias (array length = n_cycles)
      - n_cycles (int)
    """
    common_x = np.linspace(0, 100, 101)
    cycle_ids = np.unique(df_gait['cycle_id'].values)

    variables = [
        ("Knee Angle",
         "right_knee_angle_smooth", "left_knee_angle_smooth",
         "knee_angle_r", "knee_angle_l"),
        ("Hip Angle",
         "right_hip_angle_smooth", "left_hip_angle_smooth",
         "hip_flexion_r", "hip_flexion_l")
    ]

    results = {}

    for label, vid_r, vid_l, mot_r, mot_l in variables:
        # choose columns based on analyze_side
        def pick(col_r, col_l):
            if analyze_side == "right": return col_r
            if analyze_side == "left": return col_l
            if analyze_side == "both": return (col_r, col_l)
            raise ValueError("analyze_side must be 'right', 'left' or 'both'.")

        col_video = pick(vid_r, vid_l)
        col_mot = pick(mot_r, mot_l)

        # validate columns
        def exists(c):
            if isinstance(c, tuple):
                return all([cc in df_gait.columns for cc in c])
            return c in df_gait.columns

        if not exists(col_video) or not exists(col_mot):
            print(f"[compare_gait_methods] Missing columns for {label}; skipping.")
            continue

        video_cycles = []
        mot_cycles = []
        per_cycle_mae = []
        per_cycle_bias = []

        for cid in cycle_ids:
            subset = df_gait[df_gait['cycle_id'] == cid]
            if len(subset) < 2:
                continue

            # interpolation helper
            def interp_col(col):
                if isinstance(col, tuple):
                    f1 = interpolate.interp1d(subset['time_norm'], subset[col[0]],
                                              kind='linear', fill_value='extrapolate')
                    f2 = interpolate.interp1d(subset['time_norm'], subset[col[1]],
                                              kind='linear', fill_value='extrapolate')
                    return 0.5 * (f1(common_x) + f2(common_x))
                else:
                    f = interpolate.interp1d(subset['time_norm'], subset[col],
                                             kind='linear', fill_value='extrapolate')
                    return f(common_x)

            v = interp_col(col_video)
            m = interp_col(col_mot)
            video_cycles.append(v)
            mot_cycles.append(m)

            per_cycle_mae.append(np.mean(np.abs(v - m)))
            per_cycle_bias.append(np.mean(v - m))

        if len(video_cycles) == 0:
            print(f"[compare_gait_methods] No valid cycles for {label}; skipping.")
            continue

        video_cycles = np.vstack(video_cycles)  # n_cycles x 101
        mot_cycles = np.vstack(mot_cycles)

        # mean & sd
        mean_video = np.mean(video_cycles, axis=0)
        mean_mot = np.mean(mot_cycles, axis=0)
        sd_video = np.std(video_cycles, axis=0, ddof=1)
        sd_mot = np.std(mot_cycles, axis=0, ddof=1)

        # metrics on mean curves
        diff_curve = mean_video - mean_mot
        abs_error_curve = np.abs(diff_curve)
        MAE_curve = np.mean(abs_error_curve)
        mean_bias = np.mean(diff_curve)

        # pearson on mean curves
        try:
            pearson_r, pearson_p = stats.pearsonr(mean_video, mean_mot)
        except Exception:
            pearson_r, pearson_p = np.nan, np.nan

        # cross-correlation max normalized
        cross_corr_max = max_normalized_crosscorr(mean_video, mean_mot)

        # ICC across timepoints (treat each timepoint as subject, two raters = mean_video & mean_mot)
        stacked_time = np.vstack([mean_video, mean_mot]).T  # 101 x 2
        icc_time = ICC2_1_matrix(stacked_time)

        # ICC across cycles: compute per-cycle mean value for each cycle for video and mot
        per_cycle_mean_video = np.mean(video_cycles, axis=1)  # n_cycles
        per_cycle_mean_mot = np.mean(mot_cycles, axis=1)
        stacked_cycles = np.vstack([per_cycle_mean_video, per_cycle_mean_mot]).T  # n_cycles x 2
        icc_cycles = ICC2_1_matrix(stacked_cycles)

        results[label] = {
            "mean_video": mean_video,
            "mean_mot": mean_mot,
            "sd_video": sd_video,
            "sd_mot": sd_mot,
            "abs_error_curve": abs_error_curve,
            "MAE_curve": MAE_curve,
            "mean_bias": mean_bias,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "cross_corr_max": cross_corr_max,
            "ICC_time": icc_time,
            "ICC_cycles": icc_cycles,
            "per_cycle_MAE": np.array(per_cycle_mae),
            "per_cycle_bias": np.array(per_cycle_bias),
            "n_cycles": video_cycles.shape[0],
            "common_x": common_x
        }

    return results

# ---------------------------
# 9) Plotting functions 
# ---------------------------
def _save_fig(fig, path, dpi=200):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_mean_sd_overlay(mean_video, sd_video, mean_mot, sd_mot, common_x, title, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 4.5))  # fits A4 nicely when inserted
    ax.plot(common_x, mean_video, label='Video Mean', linewidth=2)
    ax.fill_between(common_x, mean_video - sd_video, mean_video + sd_video, alpha=0.25)
    ax.plot(common_x, mean_mot, label='MOT Mean', linewidth=2)
    ax.fill_between(common_x, mean_mot - sd_mot, mean_mot + sd_mot, alpha=0.25)
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel('Angle (°)')
    ax.set_title(title + " — Mean ± SD")
    ax.legend()
    ax.grid(alpha=0.4, linestyle='--')
    return _save_fig(fig, out_path)

def plot_abs_error_curve(abs_error_curve, common_x, title, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.plot(common_x, abs_error_curve, linewidth=2)
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel('Absolute Error (°)')
    ax.set_title(title + ' — Absolute Error Curve')
    ax.grid(alpha=0.4, linestyle='--')
    return _save_fig(fig, out_path)

def plot_bland_altman(mean_video, mean_mot, title, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    mean_vals = 0.5 * (mean_video + mean_mot)
    diff = mean_video - mean_mot
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    ax.scatter(mean_vals, diff, alpha=0.6)
    ax.axhline(md, color='red', linestyle='-')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_xlabel('Mean of Methods (°)')
    ax.set_ylabel('Difference (Video - MOT) (°)')
    ax.set_title(title + ' — Bland–Altman')
    ax.grid(alpha=0.3, linestyle='--')
    # Annotate bias and limits
    ax.text(0.02, 0.95, f"Bias = {md:.3f}\n±1.96 SD = [{md-1.96*sd:.3f}, {md+1.96*sd:.3f}]",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.4))
    return _save_fig(fig, out_path)

def plot_mae_boxplot(per_cycle_MAE, title, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.boxplot(per_cycle_MAE, vert=True, widths=0.5, patch_artist=True)
    ax.set_ylabel('MAE per cycle (°)')
    ax.set_title(title + ' — Per-cycle MAE (boxplot)')
    ax.grid(alpha=0.3, linestyle='--')
    return _save_fig(fig, out_path)

def plot_scatter_mot_vs_video(mean_mot, mean_video, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(mean_mot, mean_video, alpha=0.7)
    # regression line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_mot, mean_video)
        xs = np.linspace(np.min(mean_mot), np.max(mean_mot), 100)
        ax.plot(xs, slope*xs + intercept, linestyle='-', linewidth=2, label=f"y={slope:.3f}x+{intercept:.3f}\nR={r_value:.3f}")
    except Exception:
        r_value = np.nan
    ax.set_xlabel('MOT')
    ax.set_ylabel('Video')
    ax.set_title(title + ' — MOT vs Video')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return _save_fig(fig, out_path)


# ---------------------------
# 10) PDF Report generator
# ---------------------------
def generate_gait_report(results, pdf_path, metadata=None):
    """
    Generate compact PDF report (Option 2 style) using ReportLab.
    `results` is the dict returned by compare_gait_methods.
    `pdf_path` is full path to output PDF.
    `metadata` is optional dict: e.g. {'subject': 'S01', 'side': 'right', 'date': '...'}
    """
    if metadata is None:
        metadata = {}

    tmpdir = tempfile.mkdtemp(prefix="gait_report_")
    images = {}

    # Create all plots for each label
    for label, res in results.items():
        c = res['common_x']
        # filepaths
        images[label] = {}
        images[label]['mean_sd'] = os.path.join(tmpdir, f"{label}_mean_sd.png")
        images[label]['abs_err'] = os.path.join(tmpdir, f"{label}_abs_err.png")
        images[label]['ba'] = os.path.join(tmpdir, f"{label}_bland_altman.png")
        images[label]['box'] = os.path.join(tmpdir, f"{label}_mae_box.png")
        images[label]['scatter'] = os.path.join(tmpdir, f"{label}_scatter.png")

        # generate
        plot_mean_sd_overlay(res['mean_video'], res['sd_video'],
                             res['mean_mot'], res['sd_mot'], c, label, images[label]['mean_sd'])

        plot_abs_error_curve(res['abs_error_curve'], c, label, images[label]['abs_err'])
        plot_bland_altman(res['mean_video'], res['mean_mot'], label, images[label]['ba'])
        plot_mae_boxplot(res['per_cycle_MAE'], label, images[label]['box'])
        plot_scatter_mot_vs_video(res['mean_mot'], res['mean_video'], label, images[label]['scatter'])

    # Build PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story = []

    # Title Page (compact)
    title_text = metadata.get('title', 'Compact Gait Comparison Report')
    story.append(Paragraph(f"<b>{title_text}</b>", styles['Title']))
    story.append(Spacer(1, 6))
    info_lines = []
    info_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if 'subject' in metadata: info_lines.append(f"Subject: {metadata['subject']}")
    if 'side' in metadata: info_lines.append(f"Side: {metadata['side']}")
    if 'notes' in metadata: info_lines.append(f"Notes: {metadata['notes']}")
    for ln in info_lines:
        story.append(Paragraph(ln, styles['Normal']))
    story.append(Spacer(1, 8))

    # Summary Table (one row per label)
    table_data = [["Measure", "N cycles", "MAE (°)", "Bias (°)", "ICC_time", "ICC_cycles", "Pearson r", "Cross-corr"]]
    for label, res in results.items():
        table_data.append([
            label,
            f"{res['n_cycles']}",
            f"{res['MAE_curve']:.3f}",
            f"{res['mean_bias']:.3f}",
            f"{res['ICC_time']:.3f}" if not np.isnan(res['ICC_time']) else "NaN",
            f"{res['ICC_cycles']:.3f}" if not np.isnan(res['ICC_cycles']) else "NaN",
            f"{res['pearson_r']:.3f}" if not np.isnan(res['pearson_r']) else "NaN",
            f"{res['cross_corr_max']:.3f}" if not np.isnan(res['cross_corr_max']) else "NaN",
        ])

    t = Table(table_data, colWidths=[60*mm, 18*mm, 20*mm, 20*mm, 22*mm, 24*mm, 18*mm, 24*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 0.4, colors.grey),
        ('ALIGN',(1,0),(-1,-1),'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # For each label add plots: (scatter + BA) on one page, (mean±sd + abs err) on next, boxplot at bottom
    for label, res in results.items():
        # page header
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>{label}</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

        # Row: scatter (left) and BA (right)
        im_scatter = Image(images[label]['scatter'], width=90*mm, height=65*mm)
        im_ba = Image(images[label]['ba'], width=90*mm, height=65*mm)
        # Put them in a simple table for side-by-side placement
        story.append(Table([[im_scatter, im_ba]], colWidths=[95*mm, 95*mm]))
        story.append(Spacer(1, 8))

        # Row: mean±sd + abs error
        im_mean = Image(images[label]['mean_sd'], width=95*mm, height=65*mm)
        im_abs = Image(images[label]['abs_err'], width=95*mm, height=65*mm)
        story.append(Table([[im_mean, im_abs]], colWidths=[95*mm, 95*mm]))
        story.append(Spacer(1, 8))

        # Boxplot full width
        im_box = Image(images[label]['box'], width=170*mm, height=60*mm)
        story.append(im_box)

        story.append(PageBreak())

    # Final notes page
    story.append(Paragraph("<b>Notes</b>", styles['Heading2']))
    story.append(Paragraph("This compact report contains summary metrics and plots comparing Video-derived kinematics from Keypoint Detection to MOT file kinematics from OpenCap. "
                           "MAE = mean absolute error (averaged across cycle percent), bias = mean(video - mot). ICC_time treats each gait-percent point as a subject; ICC_cycles uses per-cycle means.", styles['Normal']))
    story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story)

    # cleanup temp images (optional)
    # for fdict in images.values():
    #     for p in fdict.values():
    #         try: os.remove(p)
    #         except: pass
    # os.rmdir(tmpdir)

    return pdf_path

# -----------------------------
# 11) Main workflow
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
        pdf_path = generate_gait_report(comparison_results, pdf_path="gait_compact_report.pdf",
        metadata={"subject":"S01","side":"right","title":"Subject S01 Gait Comparison"})
        print("\n📄 Generated gait comparison report:", pdf_path)


# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    main()
