#!/usr/bin/env python3
"""
Complete script:
- Ask user to pick two files
- Robust loader for comma/semicolon/tab/whitespace-delimited files (decimal point)
- Show raw signals in two separate figures
- Let user select START and END on each figure independently
- Normalize each selected segment to 0-100% (resample to equal length)
- Show normalized comparison, run statistics, save plots, export PDF
"""

import os
import io
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import scipy.stats as stats
import pingouin as pg
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# Optional: force interactive GUI backend (uncomment if needed)
# plt.switch_backend("TkAgg")

# Change cwd to script location (if available)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except Exception:
    pass


# -------------------------
# Robust loader for files
# -------------------------
def load_txt_data(filepath):
    """
    Load time and right-knee signal from a file.
    Tries several delimiters (comma, semicolon, tab, whitespace).
    Expects decimal point (.) in numbers.
    Returns: t (1D numpy), knee (1D numpy)
    Prints first 5 rows for debugging.
    """
    # Read raw text for inspection
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Try pandas auto-detect first (engine='python' sniffing)
    buf = io.StringIO(text)
    df = None
    try:
        df = pd.read_csv(buf, sep=None, engine="python")
    except Exception:
        # will attempt manual delimiters below
        df = None

    if df is None or df.shape[1] == 1:
        # Try explicit delimiters in order of likelihood
        delims = [",", ";", "\t", r"\s+"]
        for d in delims:
            try:
                buf = io.StringIO(text)
                df_try = pd.read_csv(buf, sep=d, engine="python")
                # require at least two columns
                if df_try.shape[1] >= 2:
                    df = df_try
                    break
            except Exception:
                continue

    if df is None:
        # Last resort: whitespace split
        buf = io.StringIO(text)
        df = pd.read_csv(buf, sep=r"\s+", engine="python", header=None)

    # If header looks numeric (pandas used first line as header but they are numeric),
    # re-read without header
    colnames = list(df.columns)
    if all(re.fullmatch(r"[-+]?\d+(\.\d+)?", str(c)) for c in colnames):
        # re-read without header
        buf = io.StringIO(text)
        df = pd.read_csv(buf, sep=None, engine="python", header=None)

    # Normalize column names to strings and lowercase for matching
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower = [c.lower() for c in df.columns]

    # Debug print: show first 5 parsed rows and columns
    print(f"\n--- Parsed '{os.path.basename(filepath)}' ---")
    print("Detected columns:", df.columns.tolist())
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
        try:
            print(df.head(5))
        except Exception:
            print("(Could not print preview)")

    # If only two columns assume (time, right_knee)
    if df.shape[1] == 2:
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        knee = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
    else:
        # Find right-knee column by common names
        knee_candidates = [
            "knee_angle_r", "knee_r", "right_knee", "right_knee_angle",
            "knee_angle_right", "knee_angle_r_smooth", "right_knee_angle_smooth"
        ]
        knee_col = None
        for cand in knee_candidates:
            for i, col in enumerate(cols_lower):
                if cand in col:
                    knee_col = df.columns[i]
                    break
            if knee_col:
                break

        # fuzzy fallback: column containing 'knee' and ('r' or 'right')
        if knee_col is None:
            for i, col in enumerate(cols_lower):
                if "knee" in col and ("r" in col or "right" in col):
                    knee_col = df.columns[i]
                    break

        # final fallback: any column that has 'knee' in its name
        if knee_col is None:
            for i, col in enumerate(cols_lower):
                if "knee" in col:
                    knee_col = df.columns[i]
                    break

        if knee_col is None:
            # If no knee column found, raise informative error
            raise ValueError(f"Could not detect right-knee column in '{filepath}'. Columns: {df.columns.tolist()}")

        # Detect time column
        time_candidates = ["time", "time_sec", "t", "frame", "timestamp"]
        time_col = None
        for cand in time_candidates:
            for i, col in enumerate(cols_lower):
                if cand == col or cand in col:
                    time_col = df.columns[i]
                    break
            if time_col:
                break

        if time_col is None:
            # fallback to first column
            time_col = df.columns[0]

        t = pd.to_numeric(df[time_col], errors="coerce").values
        knee = pd.to_numeric(df[knee_col], errors="coerce").values

    # Remove rows with NaN
    mask = ~np.isnan(t) & ~np.isnan(knee)
    t = t[mask]
    knee = knee[mask]

    # If still empty, raise
    if t.size == 0 or knee.size == 0:
        raise ValueError(f"No numeric data parsed from '{filepath}'. Check delimiter and decimal format.")

    return t, knee


# -------------------------
# Interactive selector class
# -------------------------
class GaitCycleSelector:
    def __init__(self, x, y, title="Select Start and End of Gait Cycle"):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.title = title
        self.coords = []

    def onclick(self, event):
        if event.inaxes is not None:
            self.coords.append(event.xdata)
            print(f"Selected x = {event.xdata:.6f}")
            # Close when have 2 points
            if len(self.coords) >= 2:
                plt.close()

    def select_cycle(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.x, self.y, label="Signal")
        ax.set_title(self.title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Knee angle [°]")
        ax.legend()
        ax.grid(True)

        cid = fig.canvas.mpl_connect("button_press_event", self.onclick)
        print(f"\n-- Please click START and END on the plot titled: '{self.title}' --")
        plt.show()

        if len(self.coords) < 2:
            raise RuntimeError("Less than 2 points selected. Please select exactly two points.")

        start, end = sorted(self.coords[:2])
        print(f"Selected interval: {start:.6f} to {end:.6f}")

        mask = (self.x >= start) & (self.x <= end)
        x_cut = self.x[mask]
        y_cut = self.y[mask]

        if x_cut.size < 2:
            raise RuntimeError("Selected region has too few data points. Try a larger region or reselect.")

        # Normalize x to 0-100% with same length as selected sample
        x_norm = np.linspace(0, 100, len(x_cut))
        return x_norm, y_cut


# -------------------------
# Statistical utilities (same as yours)
# -------------------------
def mae(a, b):
    return np.mean(np.abs(a - b))

def cross_corr(a, b):
    corr = np.correlate(a - np.mean(a), b - np.mean(b), mode='full')
    lag = np.argmax(corr) - (len(a) - 1)
    return corr, lag

def cohens_d_paired(a, b):
    d = a - b
    return np.mean(d) / np.std(d, ddof=1)

def mean_ci_95(x):
    mean = np.mean(x)
    sem = stats.sem(x)
    ci_low, ci_high = stats.t.interval(0.95, df=len(x)-1, loc=mean, scale=sem)
    return mean, ci_low, ci_high

def bland_altman(a, b):
    diff = a - b
    mean_values = (a + b) / 2
    bias = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    return mean_values, diff, bias, loa_low, loa_high

def compare_methods(a, b, name_a="Methode A", name_b="Methode B", t=None):
    if t is None:
        t = np.arange(len(a))

    mae_val = mae(a, b)
    try:
        pearson_r, pearson_p = stats.pearsonr(a, b)
    except Exception:
        pearson_r = pearson_p = np.nan

    _, lag = cross_corr(a, b)

    # ICC
    try:
        df_icc = pd.DataFrame({
            "Subject": np.repeat(np.arange(len(a)), 2),
            "Method": [name_a, name_b] * len(a),
            "Value": np.concatenate([a, b])
        })
        icc = pg.intraclass_corr(data=df_icc, targets="Subject", raters="Method", ratings="Value")
        icc_res = icc[icc["Type"].str.contains("ICC2", na=False)]
        ICC_value = float(icc_res["ICC"].iloc[0]) if not icc_res.empty else float(icc["ICC"].iloc[0])
    except Exception:
        ICC_value = np.nan

    diff = b - a
    try:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
    except Exception:
        shapiro_p = np.nan
    try:
        _, t_p = stats.ttest_rel(a, b)
    except Exception:
        t_p = np.nan
    try:
        _, w_p = stats.wilcoxon(a, b)
    except Exception:
        w_p = np.nan

    try:
        d_cohen = cohens_d_paired(a, b)
    except Exception:
        d_cohen = np.nan

    mean_a, ci_low_a, ci_high_a = mean_ci_95(a)
    mean_b, ci_low_b, ci_high_b = mean_ci_95(b)

    try:
        SD = np.std(a, ddof=1)
        SEM = SD * np.sqrt(1 - ICC_value) if not np.isnan(ICC_value) else np.nan
        MDC = 1.96 * np.sqrt(2) * SEM if not np.isnan(SEM) else np.nan
    except Exception:
        SEM = MDC = np.nan

    mean_vals, diff_vals, bias, loa_low, loa_high = bland_altman(b, a)

    results = {
        "error": {"MAE": mae_val, "bias": bias, "LOA_low": loa_low, "LOA_high": loa_high},
        "correlation": {"pearson_r": pearson_r, "pearson_p": pearson_p, "crosscorr_lag": lag, "ICC": ICC_value},
        "tests": {"shapiro_p": shapiro_p, "t_test_p": t_p, "wilcoxon_p": w_p},
        "effect_sizes": {"cohens_d": d_cohen},
        "confidence_intervals": {name_a: (mean_a, ci_low_a, ci_high_a), name_b: (mean_b, ci_low_b, ci_high_b)},
        "variation": {"SEM": SEM, "MDC": MDC},
        "bland_altman": {"mean_vals": mean_vals, "diff_vals": diff_vals}
    }
    return results


# -------------------------
# PDF export (normalized)
# -------------------------
def export_pdf_report(results, plot_path_main, plot_path_ba, filename="methodenvergleich_report_normalized.pdf"):
    styles = getSampleStyleSheet()
    story = []
    doc = SimpleDocTemplate(filename, pagesize=A4)

    story.append(Paragraph("<b>Methodenvergleich (normalisierte Gangzyklen)</b>", styles['Title']))
    story.append(Spacer(1, 0.5 * cm))

    text = f"""
    <b>Fehlermaße:</b><br/>
    MAE: {results['error']['MAE']:.3f}<br/>
    Bias: {results['error']['bias']:.3f}<br/>
    LOA-Low: {results['error']['LOA_low']:.3f}<br/>
    LOA-High: {results['error']['LOA_high']:.3f}<br/><br/>

    <b>Korrelation:</b><br/>
    Pearson r: {results['correlation']['pearson_r']:.3f}<br/>
    Pearson p: {results['correlation']['pearson_p']:.3f}<br/>
    ICC: {results['correlation']['ICC']:.3f}<br/>
    CrossCorr Lag: {results['correlation']['crosscorr_lag']} Frames<br/><br/>

    <b>Statistische Tests:</b><br/>
    Shapiro-Wilk p: {results['tests']['shapiro_p']:.3f}<br/>
    t-Test p: {results['tests']['t_test_p']:.3f}<br/>
    Wilcoxon p: {results['tests']['wilcoxon_p']:.3f}<br/><br/>

    <b>Effektstärke:</b><br/>
    Cohen's d: {results['effect_sizes']['cohens_d']:.3f}<br/><br/>

    <b>Variabilität:</b><br/>
    SEM: {results['variation']['SEM']:.3f}<br/>
    MDC: {results['variation']['MDC']:.3f}<br/>
    """

    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("<b>Verlauf + Differenzkurve (normalisiert)</b>", styles["Heading2"]))
    story.append(Image(plot_path_main, width=16*cm, height=8*cm))
    story.append(Spacer(1, 0.7*cm))

    story.append(Paragraph("<b>Bland-Altman Plot (normalisiert)</b>", styles["Heading2"]))
    story.append(Image(plot_path_ba, width=16*cm, height=8*cm))

    doc.build(story)
    print(f"PDF created: {filename}")


# -------------------------
# Main workflow
# -------------------------
def main():
    # Ask user to pick two files
    root = Tk()
    root.withdraw()
    print("Select file 1 (e.g. OpenPose output)")
    file1 = filedialog.askopenfilename(title="Select first data file")
    if not file1:
        print("No file selected. Exiting.")
        return
    print("Select file 2 (e.g. KPD output)")
    file2 = filedialog.askopenfilename(title="Select second data file")
    if not file2:
        print("No file selected. Exiting.")
        return
    root.destroy()

    # Load both files
    t1, sig1 = load_txt_data(file1)
    t2, sig2 = load_txt_data(file2)

    # Show raw signals in TWO separate figures for clarity
    fig1 = plt.figure(figsize=(10, 3.5))
    plt.plot(t1, sig1, label=os.path.basename(file1))
    plt.title("Raw signal 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee angle [°]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(10, 3.5))
    plt.plot(t2, sig2, label=os.path.basename(file2))
    plt.title("Raw signal 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Knee angle [°]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Interactive selection (separate for each dataset)
    selector1 = GaitCycleSelector(t1, sig1, title=f"Signal 1: {os.path.basename(file1)} - select START & END")
    t1_norm, sig1_norm = selector1.select_cycle()

    selector2 = GaitCycleSelector(t2, sig2, title=f"Signal 2: {os.path.basename(file2)} - select START & END")
    t2_norm, sig2_norm = selector2.select_cycle()

    # Resample to equal length (minimum)
    N = min(len(sig1_norm), len(sig2_norm))
    if N < 3:
        raise RuntimeError("Selected segments too short for meaningful analysis.")
    sig1_rs = np.interp(np.linspace(0, len(sig1_norm)-1, N), np.arange(len(sig1_norm)), sig1_norm)
    sig2_rs = np.interp(np.linspace(0, len(sig2_norm)-1, N), np.arange(len(sig2_norm)), sig2_norm)
    t_norm = np.linspace(0, 100, N)

    # Show normalized comparison for verification
    plt.figure(figsize=(10, 4))
    plt.plot(t_norm, sig1_rs, label="Signal 1 (normalized)")
    plt.plot(t_norm, sig2_rs, label="Signal 2 (normalized)")
    plt.xlabel("Gait cycle [%]")
    plt.ylabel("Knee angle [°]")
    plt.title("Normalized gait cycles (verification)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Run comparison on normalized
    results = compare_methods(sig1_rs, sig2_rs, "Signal1", "Signal2", t_norm)

    # Save normalized comparison plot (main)
    main_plot_path = "plot_normalized_comparison.png"
    plt.figure(figsize=(12, 6))
    plt.plot(t_norm, sig1_rs, label="Signal 1 (normalized)")
    plt.plot(t_norm, sig2_rs, label="Signal 2 (normalized)")
    plt.plot(t_norm, sig2_rs - sig1_rs, label="Difference (2 - 1)", linestyle="--")
    plt.xlabel("Gait cycle [%]")
    plt.ylabel("Knee angle [°]")
    plt.title("Normalized Comparison: Signal1 vs Signal2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(main_plot_path, dpi=200)
    plt.close()

    # Bland-Altman normalized
    mean_vals = results["bland_altman"]["mean_vals"]
    diff_vals = results["bland_altman"]["diff_vals"]
    bias = results["error"]["bias"]
    loa_low = results["error"]["LOA_low"]
    loa_high = results["error"]["LOA_high"]

    ba_plot_path = "plot_bland_altman_normalized.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.7)
    plt.axhline(bias, label=f"Bias = {bias:.2f}")
    plt.axhline(loa_low, linestyle="--", label="LOA Low")
    plt.axhline(loa_high, linestyle="--", label="LOA High")
    plt.xlabel("Mean [°]")
    plt.ylabel("Difference (2 - 1) [°]")
    plt.title("Bland-Altman (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ba_plot_path, dpi=200)
    plt.close()

    # Print results
    print("\n=== Comparison results (normalized) ===")
    print(f"MAE: {results['error']['MAE']:.4f}")
    print(f"Bias: {results['error']['bias']:.4f}")
    print(f"LOA: [{results['error']['LOA_low']:.4f}, {results['error']['LOA_high']:.4f}]")
    print(f"Pearson r: {results['correlation']['pearson_r']:.4f} (p={results['correlation']['pearson_p']})")
    print(f"ICC: {results['correlation']['ICC']}")
    print("======================================\n")

    # Export PDF
    export_pdf_report(results, main_plot_path, ba_plot_path)

    print("Done.")

if __name__ == "__main__":
    main()
