import os, json, math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def to_deg_if_rad(a):
    """Auto-detect radians vs degrees; return degrees."""
    a = np.asarray(a)
    if np.nanmax(np.abs(a)) <= math.pi + 1e-3:
        return a * (180.0 / math.pi)
    return a

def great_circle_error_deg(gt_pitch, gt_yaw, pr_pitch, pr_yaw):
    """
    Angular error between two directions on a sphere (degrees).
    Uses spherical law of cosines on (pitch, yaw) in radians (internally).
    Pitch θ (down +), yaw ψ (left +).
    """
    # Convert to radians for stable trig, regardless of input unit
    gth, gty = np.radians(to_deg_if_rad(gt_pitch)), np.radians(to_deg_if_rad(gt_yaw))
    pth, pty = np.radians(to_deg_if_rad(pr_pitch)), np.radians(to_deg_if_rad(pr_yaw))

    # spherical law of cosines:
    # cos(err) = sin θ1 sin θ2 + cos θ1 cos θ2 cos(ψ1 - ψ2)
    cos_err = np.sin(gth)*np.sin(pth) + np.cos(gth)*np.cos(pth)*np.cos(gty - pty)
    cos_err = np.clip(cos_err, -1.0, 1.0)
    err = np.degrees(np.arccos(cos_err))
    return err

def wrap180_deg(a):
    a = (a + 180.0) % 360.0 - 180.0
    return a

def load_csv(path, prefer_calib=False):
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    # choose prediction columns (baseline wants raw pr_*)
    pr_p, pr_y = None, None
    if prefer_calib:
        if {'pr_pitch_calib','pr_yaw_calib'}.issubset(cols):
            pr_p, pr_y = 'pr_pitch_calib', 'pr_yaw_calib'
    if pr_p is None or pr_y is None:
        if {'pr_pitch','pr_yaw'}.issubset(cols):
            pr_p, pr_y = 'pr_pitch', 'pr_yaw'
        else:
            raise ValueError("Could not find prediction columns in CSV (need pr_pitch/pr_yaw).")

    if not {'gt_pitch','gt_yaw'}.issubset(cols):
        raise ValueError("CSV must contain gt_pitch and gt_yaw.")

    # cast to float
    df['gt_pitch'] = df['gt_pitch'].astype(float)
    df['gt_yaw']   = df['gt_yaw'].astype(float)
    df[pr_p] = df[pr_p].astype(float)
    df[pr_y] = df[pr_y].astype(float)

    return df, pr_p, pr_y

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Plots
# -----------------------------
def plot_hist(err_deg, out_png):
    plt.figure(figsize=(6,4))
    bins = np.arange(0, 91, 2)  # 0..90 deg in 2-deg steps
    plt.hist(err_deg, bins=bins, edgecolor='black', linewidth=0.5)
    median = np.median(err_deg)
    plt.axvline(median, linestyle='--', linewidth=1.5, label=f"median={median:.1f}°")
    plt.xlabel("Error (deg)")
    plt.ylabel("Count")
    plt.title("Angular-error histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_cdf(err_deg, out_png, markers=(5,10,15,20)):
    x = np.sort(err_deg)
    y = np.linspace(0, 1, len(x), endpoint=True)
    plt.figure(figsize=(6,4))
    plt.plot(x, y, linewidth=2)
    for k in markers:
        # fraction <= k
        frac = (err_deg <= k).mean() if len(err_deg) else np.nan
        plt.axvline(k, color='gray', linestyle='--', linewidth=1)
        plt.text(k, 0.02, f"{k}°", rotation=90, va='bottom', ha='right', fontsize=8)
    plt.xlabel("Error (deg)")
    plt.ylabel("CDF")
    plt.title("Angular-error CDF")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def pose_heatmap(gt_pitch_deg, gt_yaw_deg, err_deg, out_png):
    # Bin on absolute yaw/pitch magnitudes
    abs_p = np.abs(gt_pitch_deg)
    abs_y = np.abs(gt_yaw_deg)
    # Define bin edges (simple, readable)
    p_edges = [0, 10, 20, 30, 45, 60, 90]
    y_edges = [0, 10, 20, 30, 45, 60, 90]

    H = np.full((len(p_edges)-1, len(y_edges)-1), np.nan)
    for i in range(len(p_edges)-1):
        for j in range(len(y_edges)-1):
            mask = (abs_p >= p_edges[i]) & (abs_p < p_edges[i+1]) & \
                   (abs_y >= y_edges[j]) & (abs_y < y_edges[j+1])
            if np.any(mask):
                H[i,j] = np.nanmean(err_deg[mask])

    plt.figure(figsize=(6,5))
    im = plt.imshow(H, origin='lower', aspect='auto',
                    extent=[y_edges[0], y_edges[-1], p_edges[0], p_edges[-1]],
                    cmap='magma')
    cbar = plt.colorbar(im)
    cbar.set_label("MAE (deg)")
    plt.xlabel("|yaw| (deg)")
    plt.ylabel("|pitch| (deg)")
    plt.title("Per-pose MAE heatmap (baseline)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def mae_vs_abs_axis(abs_axis_deg, err_deg, labels, out_png):
    # centers in simple bins
    edges = [0, 10, 20, 30, 45, 60, 90]
    centers = []
    maes = []
    for i in range(len(edges)-1):
        mask = (abs_axis_deg >= edges[i]) & (abs_axis_deg < edges[i+1])
        if np.any(mask):
            centers.append(0.5*(edges[i]+edges[i+1]))
            maes.append(np.nanmean(err_deg[mask]))
    plt.figure(figsize=(6,4))
    plt.plot(centers, maes, marker='o')
    plt.xlabel(labels['x'])
    plt.ylabel("MAE (deg)")
    plt.title(labels['title'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False,
                    default="/users/project1/pt01281/gaze_outputs/preds_test_gaze_exacthead_stride1.csv",
                    help="Baseline ETH test CSV (gt_pitch/gt_yaw & pr_pitch/pr_yaw).")
    ap.add_argument("--outdir", required=False,
                    default="/users/project1/pt01281/gaze_outputs/eth_eval",
                    help="Output dir for metrics and figs/{...}.")
    ap.add_argument("--prefer-calib", action="store_true",
                    help="Use pr_*_calib if present (off by default for raw baseline).")
    args = ap.parse_args()

    print("[LOAD]", args.csv)
    df, prp_col, pry_col = load_csv(args.csv, prefer_calib=args.prefer_calib)

    gt_p_deg = to_deg_if_rad(df['gt_pitch'].to_numpy())
    gt_y_deg = to_deg_if_rad(df['gt_yaw'].to_numpy())
    pr_p_deg = to_deg_if_rad(df[prp_col].to_numpy())
    pr_y_deg = to_deg_if_rad(df[pry_col].to_numpy())

    # Wrap residual axes for reporting (not used by the spherical error)
    res_p = wrap180_deg(pr_p_deg - gt_p_deg)
    res_y = wrap180_deg(pr_y_deg - gt_y_deg)

    # Angular error
    err_deg = great_circle_error_deg(df['gt_pitch'].to_numpy(),
                                     df['gt_yaw'].to_numpy(),
                                     df[prp_col].to_numpy(),
                                     df[pry_col].to_numpy())

    # Metrics
    mae_avg = float(np.nanmean(err_deg))
    mae_pitch = float(np.nanmean(np.abs(res_p)))
    mae_yaw   = float(np.nanmean(np.abs(res_y)))

    # within-K
    def within(k): return float(np.mean(err_deg <= k)) if len(err_deg) else float('nan')
    within_points = [5, 10, 15, 20]
    within_rows = [{"K_deg": k, "within_frac": within(k)} for k in within_points]

    # Write out
    ensure_dir(args.outdir)
    figs_dir = os.path.join(args.outdir, "figs")
    ensure_dir(figs_dir)

    metrics_path = os.path.join(args.outdir, "eth_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mae_avg_deg": round(mae_avg, 4),
            "mae_pitch_deg": round(mae_pitch, 4),
            "mae_yaw_deg": round(mae_yaw, 4),
            "N": int(len(df))
        }, f, indent=2)
    print("[WRITE]", metrics_path)

    within_path = os.path.join(args.outdir, "eth_withinK.csv")
    pd.DataFrame(within_rows).to_csv(within_path, index=False)
    print("[WRITE]", within_path)

    # Plots
    plot_hist(err_deg, os.path.join(figs_dir, "F1_eth_err_hist.png"))
    plot_cdf(err_deg, os.path.join(figs_dir, "F2_eth_err_cdf.png"))
    pose_heatmap(gt_p_deg, gt_y_deg, err_deg, os.path.join(figs_dir, "F3_eth_pose_heatmap.png"))
    mae_vs_abs_axis(np.abs(gt_y_deg), err_deg,
                    {"x":"|yaw| (deg)", "title":"MAE vs |yaw|"},
                    os.path.join(figs_dir, "F4a_mae_vs_abs_yaw.png"))
    mae_vs_abs_axis(np.abs(gt_p_deg), err_deg,
                    {"x":"|pitch| (deg)", "title":"MAE vs |pitch|"},
                    os.path.join(figs_dir, "F4b_mae_vs_abs_pitch.png"))

    print("[DONE] figs in", figs_dir)
