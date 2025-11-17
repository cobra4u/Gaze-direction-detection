import os, argparse, re, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TO_DEG = 180/np.pi
def wrap180(a): return ((a+180.0)%360.0)-180.0

def extract_sid_from_path(s):
    if not isinstance(s, str): return None
    m = re.search(r"(subject\d{4})", s)
    return m.group(1) if m else None

def load_preds(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure subject_id
    if "subject_id" not in df.columns:
        if "subject" in df.columns:
            df = df.rename(columns={"subject":"subject_id"})
        else:
            df["subject_id"] = df.get("path","").astype(str).map(extract_sid_from_path).fillna("unknown")
    need = {"gt_pitch","gt_yaw","pr_pitch","pr_yaw","subject_id"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[{os.path.basename(csv_path)}] missing columns: {miss}")
    return df

def compute_errors(df, mode="raw"):
    """
    Returns err_pitch_deg, err_yaw_deg, err_avg_deg
    mode = "raw" uses pr_* vs gt_*.
    mode = "champ" prefers pitch_corr/yaw_corr else pr_*_calib else pr_*.
    """
    gp = df["gt_pitch"].astype(float)
    gy = df["gt_yaw"].astype(float)

    def pick_cols(preferred_pairs):
        for a,b in preferred_pairs:
            if a in df.columns and b in df.columns:
                return df[a].astype(float), df[b].astype(float)
        return None, None

    if mode == "raw":
        pp, py = df["pr_pitch"].astype(float), df["pr_yaw"].astype(float)
    else:
        # champion / calibrated fallback chain
        pp, py = pick_cols([("pitch_corr","yaw_corr"),
                            ("pr_pitch_calib","pr_yaw_calib"),
                            ("pr_pitch","pr_yaw")])
        if pp is None:  # ultimate fallback
            pp, py = df["pr_pitch"].astype(float), df["pr_yaw"].astype(float)

    ep = wrap180((pp - gp) * TO_DEG)
    ey = wrap180((py - gy) * TO_DEG)
    ea = (ep.abs() + ey.abs())/2.0
    return ep, ey, ea

def ecdf(values):
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0: return v, np.array([])
    y = np.arange(1, n+1) / n
    return v, y

def plot_cdf_overlay(raw_err_deg, champ_err_deg, out_png):
    plt.figure(figsize=(6,4))
    x1, y1 = ecdf(raw_err_deg)
    x2, y2 = ecdf(champ_err_deg)
    plt.plot(x1, y1, label="Raw baseline")
    plt.plot(x2, y2, label="Calibrated (champion)")
    for K in (5,10,15,20):
        plt.axvline(K, linestyle="--", linewidth=0.8, alpha=0.5)
    plt.xlabel("Angular error (deg)")
    plt.ylabel("CDF")
    plt.title("ETH-XGaze: CDF of angular error")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[WRITE] {out_png}")

def load_demo_csv(path):
    if not path or not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    # normalize column names
    if "subject_id" not in d.columns:
        for k in ("subject","sid"):
            if k in d.columns: d = d.rename(columns={k:"subject_id"})
    if "gender" not in d.columns:
        if "sex" in d.columns: d = d.rename(columns={"sex":"gender"})
    if "age_bin" not in d.columns:
        for k in ("age_group","agebin","ageBin"):
            if k in d.columns: d = d.rename(columns={k:"age_bin"})
    # sanity defaults
    for c in ("subject_id","gender","age_bin"):
        if c not in d.columns:
            print(f"[WARN] demographics csv missing {c} -> skipping subgroup plots")
            return None
    d["gender"]  = d["gender"].fillna("unknown").astype(str)
    d["age_bin"] = d["age_bin"].fillna("unknown").astype(str)
    return d[["subject_id","gender","age_bin"]].drop_duplicates()

def group_mae(df_merged, group_col, mode_label):
    # Compute per-group mean of |err| average
    g = (df_merged.groupby(group_col, dropna=False)
                    .agg(MAE_avg_deg=("err_avg_deg", "mean"),
                         N=("err_avg_deg","size"))
                    .reset_index())
    g["mode"] = mode_label
    return g

def barplot_group_compare(dfA, dfB, group_col, labelA, labelB, out_png, title):
    # Outer-join on groups to keep union
    G = pd.merge(dfA[[group_col,"MAE_avg_deg","N"]].rename(columns={"MAE_avg_deg":labelA, "N":f"N_{labelA}"}),
                 dfB[[group_col,"MAE_avg_deg","N"]].rename(columns={"MAE_avg_deg":labelB, "N":f"N_{labelB}"}),
                 on=group_col, how="outer")
    # order groups: fixed heuristics
    if group_col == "gender":
        order = ["female","male","unknown"]
    elif group_col == "age_bin":
        order = ["18-34","35-54","55+","unknown"]
    else:
        order = sorted(G[group_col].astype(str).fillna("unknown").unique().tolist())
    G[group_col] = pd.Categorical(G[group_col], order, ordered=True)
    G = G.sort_values(group_col)

    x = np.arange(len(G))
    w = 0.38
    plt.figure(figsize=(7,4))
    plt.bar(x - w/2, G[labelA], width=w, label=labelA)
    plt.bar(x + w/2, G[labelB], width=w, label=labelB)
    plt.xticks(x, [str(v) for v in G[group_col]])
    plt.ylabel("MAE (deg)")
    plt.title(title)
    plt.legend()
    # Optional: annotate Ns under x-ticks
    for i,(nA,nB) in enumerate(zip(G.get(f"N_{labelA}",[None]*len(G)),
                                   G.get(f"N_{labelB}",[None]*len(G)))):
        s = ""
        if pd.notna(nA): s += f"N{labelA[0]}={int(nA)}"
        if pd.notna(nB): s += (", " if s else "") + f"N{labelB[0]}={int(nB)}"
        if s:
            plt.text(i, 0, s, ha="center", va="bottom", fontsize=8, rotation=90)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[WRITE] {out_png}")

def scan_candidates(base_dir):
    # Candidate calibrated CSVs to consider for the ablation bars
    cands = [
      ("raw baseline",                    "/users/project1/pt01281/gaze_outputs/preds_test_gaze_exacthead_stride1.csv"),
      ("piecewise baseline",              "/users/project1/pt01281/gaze_outputs/eth_eval_calib/preds_test_CALIB_piecewise.csv"),
      ("vec-affine",                      "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_vec_affine.csv"),
      ("poly2d",                          "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_poly2d.csv"),
      ("BLEND",                           "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND.csv"),
      ("BLEND_OFFGBIN2",                  "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND_OFFGBIN2.csv"),
      ("GRID2D_SOFT",                     "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND_OFFGBIN2_GRID2D_SOFT.csv"),
      ("ISOsoft_b0.7",                    "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND_OFFGBIN2_ISOsoft_b0.7.csv"),
      ("CHAMPION",                        "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_CHAMPION.csv"),
    ]
    return [(lab, p) for (lab,p) in cands if os.path.exists(p)]

def overall_avg_mae(csv_path):
    df = load_preds(csv_path)
    # auto detect if "champion-like"
    champish = any(c in df.columns for c in ("pitch_corr","pr_pitch_calib"))
    _, _, ea = compute_errors(df, mode=("champ" if champish else "raw"))
    return float(np.mean(np.abs(ea)))

def plot_ablation_bars(out_png):
    items = scan_candidates("/users/project1/pt01281/gaze_outputs/eth_eval_calib")
    if not items:
        print("[SKIP] No calibrated CSVs found for ablation bars.")
        return
    labels, maes = [], []
    for lab, p in items:
        try:
            m = overall_avg_mae(p)
            labels.append(lab); maes.append(m)
        except Exception as e:
            print(f"[WARN] skip {lab}: {e}")
    if not labels: 
        print("[SKIP] nothing to plot for ablations")
        return
    # order: raw baseline first if present, CHAMPION last if present
    order_idx = list(range(len(labels)))
    try:
        i_raw = labels.index("raw baseline")
        order_idx.remove(i_raw); order_idx = [i_raw] + order_idx
    except ValueError: pass
    try:
        i_ch = labels.index("CHAMPION")
        order_idx.remove(i_ch); order_idx = order_idx + [i_ch]
    except ValueError: pass
    labels = [labels[i] for i in order_idx]
    maes   = [maes[i]   for i in order_idx]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(labels)), maes)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("MAE (deg)")
    plt.title("ETH test: ablation summary (lower is better)")
    for i,v in enumerate(maes):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[WRITE] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv",   required=True)
    ap.add_argument("--champ_csv", required=True)
    ap.add_argument("--demo_csv",  default="")  # optional
    ap.add_argument("--outdir",    required=True)
    args = ap.parse_args()

    raw = load_preds(args.raw_csv)
    ch  = load_preds(args.champ_csv)

    # errors
    _, _, raw_avg = compute_errors(raw, "raw")
    _, _, ch_avg  = compute_errors(ch, "champ")

    # F10: CDF overlay
    out10 = os.path.join(args.outdir, "figs", "F10_cdf_overlay_raw_vs_champ.png")
    plot_cdf_overlay(raw_avg, ch_avg, out10)

    # Subgroups (if demographics present)
    demo = load_demo_csv(args.demo_csv)
    if demo is not None and len(demo):
        raw_m = raw.merge(demo, on="subject_id", how="left")
        ch_m  = ch.merge(demo,  on="subject_id", how="left")
        # fill unknowns to keep groups explicit
        for d in (raw_m, ch_m):
            d["gender"]  = d["gender"].fillna("unknown")
            d["age_bin"] = d["age_bin"].fillna("unknown")

        # compute errors for merged copies
        for d, mode in ((raw_m,"raw"), (ch_m,"champ")):
            ep, ey, ea = compute_errors(d, mode)
            d["err_avg_deg"] = ea

        # F11: gender bars
        gA = group_mae(raw_m, "gender",  "Raw")
        gB = group_mae(ch_m,  "gender",  "Calibrated")
        out11 = os.path.join(args.outdir, "figs", "F11_gender_raw_vs_champ.png")
        barplot_group_compare(gA, gB, "gender", "Raw", "Calibrated", out11,
                              "ETH test: MAE by gender (raw vs. calibrated)")

        # F12: age-bin bars
        aA = group_mae(raw_m, "age_bin", "Raw")
        aB = group_mae(ch_m,  "age_bin", "Calibrated")
        out12 = os.path.join(args.outdir, "figs", "F12_agebin_raw_vs_champ.png")
        barplot_group_compare(aA, aB, "age_bin", "Raw", "Calibrated", out12,
                              "ETH test: MAE by age bin (raw vs. calibrated)")
    else:
        print("[SKIP] F11/F12 — demographics CSV not provided or unusable.")

    # F13: ablation bars
    out13 = os.path.join(args.outdir, "figs", "F13_ablation_bars.png")
    plot_ablation_bars(out13)

if __name__ == "__main__":
    main()
