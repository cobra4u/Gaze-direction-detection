import os, math, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_deg(x):
    x = np.asarray(x, dtype=float)
    mx = np.nanmax(np.abs(x))
    return np.degrees(x) if mx <= 3.2 else x

def load_csv(path):
    df = pd.read_csv(path)
    # expected columns: gt_pitch, gt_yaw, pr_pitch, pr_yaw
    need = ["gt_pitch","gt_yaw","pr_pitch","pr_yaw"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV missing columns: {miss}")
    for c in need: df[c] = to_deg(df[c].values)
    # errors (deg)
    df["err_pitch"] = (df["pr_pitch"] - df["gt_pitch"]).abs()
    df["err_yaw"]   = (df["pr_yaw"]   - df["gt_yaw"]).abs()
    df["err_avg"]   = 0.5*(df["err_pitch"] + df["err_yaw"])
    # absolute pose for binning (deg)
    df["abs_yaw"]   = df["gt_yaw"].abs()
    df["abs_pitch"] = df["gt_pitch"].abs()
    # clip to [0,90] for stable grids
    for c in ["abs_yaw","abs_pitch"]:
        df[c] = df[c].clip(lower=0, upper=90)
    return df

def mae_vs_bins(ax, x, y, bins, xlabel, ylabel):
    inds = np.digitize(x, bins, right=True)
    centers, maes, ns = [], [], []
    for b in range(1, len(bins)):
        m = (inds == b)
        if m.any():
            centers.append((bins[b-1] + bins[b])/2.)
            maes.append(float(np.nanmean(y[m])))
            ns.append(int(m.sum()))
        else:
            centers.append((bins[b-1] + bins[b])/2.)
            maes.append(np.nan)
            ns.append(0)
    ax.plot(centers, maes, marker='o', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return pd.DataFrame({"bin_center_deg":centers, "MAE_deg":maes, "N":ns})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir,"figs"), exist_ok=True)
    df = load_csv(args.csv)

    # === Figure 5.3 — per-pose heatmap (|yaw|×|pitch|) of avg MAE ===
    by = np.linspace(0, 90, args.bins+1)
    bp = np.linspace(0, 90, args.bins+1)
    H = np.full((args.bins, args.bins), np.nan)
    for i in range(args.bins):
        for j in range(args.bins):
            m = (df["abs_yaw"].between(by[i], by[i+1], inclusive="left") &
                 df["abs_pitch"].between(bp[j],bp[j+1], inclusive="left"))
            if m.any():
                H[j,i] = float(df.loc[m,"err_avg"].mean())  # rows=pitch bins, cols=yaw bins

    plt.figure(figsize=(6.2,5.2), dpi=140)
    im = plt.imshow(H, origin="lower", extent=[by[0],by[-1],bp[0],bp[-1]], aspect="auto")
    cbar = plt.colorbar(im)
    cbar.set_label("MAE (deg)")
    plt.xlabel("|yaw| (deg)")
    plt.ylabel("|pitch| (deg)")
    plt.title("Per-pose MAE (baseline)")
    plt.grid(False)
    out_hm = os.path.join(args.outdir,"figs","F3_eth_pose_heatmap.png")
    plt.tight_layout(); plt.savefig(out_hm); plt.close()

    # === Figure 5.4a — MAE vs |yaw| (avg of axis MAE) ===
    bins_ = np.linspace(0, 90, args.bins+1)
    fig, ax = plt.subplots(figsize=(6.2,4.2), dpi=140)
    tbl_yaw = mae_vs_bins(ax, df["abs_yaw"].values, df["err_avg"].values,
                          bins_, "|yaw| (deg)", "MAE (deg)")
    out_y = os.path.join(args.outdir,"figs","F4a_mae_vs_abs_yaw.png")
    plt.title("MAE vs |yaw| (baseline)")
    plt.tight_layout(); plt.savefig(out_y); plt.close()

    # === Figure 5.4b — MAE vs |pitch| (avg of axis MAE) ===
    fig, ax = plt.subplots(figsize=(6.2,4.2), dpi=140)
    tbl_p = mae_vs_bins(ax, df["abs_pitch"].values, df["err_avg"].values,
                        bins_, "|pitch| (deg)", "MAE (deg)")
    out_p = os.path.join(args.outdir,"figs","F4b_mae_vs_abs_pitch.png")
    plt.title("MAE vs |pitch| (baseline)")
    plt.tight_layout(); plt.savefig(out_p); plt.close()

    # also drop the tables next to the figures for reference
    tbl_yaw.to_csv(os.path.join(args.outdir,"F4a_mae_vs_abs_yaw.csv"), index=False)
    tbl_p.to_csv(os.path.join(args.outdir,"F4b_mae_vs_abs_pitch.csv"), index=False)

if __name__ == "__main__":
    main()
