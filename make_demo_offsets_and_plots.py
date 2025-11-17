#!/usr/bin/env python3
import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    print(f"[LOAD] {path}")
    df = pd.read_csv(path)
    req = ["subject_id","pr_pitch","pr_yaw"]
    have = set(df.columns)
    if not {"pitch_corr","yaw_corr"}.issubset(have):
        # fall back to raw preds if corr not present
        df["pitch_corr"] = df["pr_pitch"]
        df["yaw_corr"]   = df["pr_yaw"]
    return df

def norm_gender(g):
    if pd.isna(g): return "unknown"
    s = str(g).strip().lower()
    if s in ["man","male","m"]: return "male"
    if s in ["woman","female","f"]: return "female"
    return s

def norm_agebin(a):
    if pd.isna(a): return "unknown"
    return str(a).strip()

def join_demo(df, demo):
    key = "subject_id"
    if key not in demo.columns:
        raise RuntimeError(f"demographics CSV missing '{key}' column")
    demo = demo.copy()
    # allow both schemas: with age_mean or not
    for col in demo.columns:
        if col == "gender":
            demo["gender"] = demo["gender"].map(norm_gender)
        if col == "age_bin":
            demo["age_bin"] = demo["age_bin"].map(norm_agebin)
    keep = [c for c in ["subject_id","gender","age_bin"] if c in demo.columns]
    demo = demo[keep].drop_duplicates("subject_id")
    out = df.merge(demo, on="subject_id", how="left")
    out["gender"]  = out["gender"].fillna("unknown")
    out["age_bin"] = out["age_bin"].fillna("unknown")
    return out

def residuals(df):
    need = ["gt_pitch","gt_yaw","pitch_corr","yaw_corr"]
    if not set(need).issubset(df.columns):
        raise RuntimeError(f"CSV needs {need}")
    df = df.copy()
    df["res_pitch"] = df["pitch_corr"] - df["gt_pitch"]
    df["res_yaw"]   = df["yaw_corr"]   - df["gt_yaw"]
    return df

def group_means(val_df, group_col):
    G = val_df.groupby(group_col, dropna=False).agg(
        mu_pitch=("res_pitch","mean"),
        mu_yaw  =("res_yaw","mean"),
        N       =("res_pitch","size"))
    return G.reset_index()

def apply_offsets(test_df, offsets, group_col, out_cols_prefix):
    D = test_df.merge(offsets, on=group_col, how="left")
    D["mu_pitch"] = D["mu_pitch"].fillna(0.0)
    D["mu_yaw"]   = D["mu_yaw"].fillna(0.0)
    pcol = f"{out_cols_prefix}_pitch"
    ycol = f"{out_cols_prefix}_yaw"
    D[pcol] = D["pitch_corr"] - D["mu_pitch"]
    D[ycol] = D["yaw_corr"]   - D["mu_yaw"]
    return D

def mae_cols(df, pcol, ycol):
    e = np.degrees(np.abs(df[pcol] - df["gt_pitch"])) + np.degrees(np.abs(df[ycol] - df["gt_yaw"]))
    return (np.degrees(np.abs(df[pcol] - df["gt_pitch"])).mean(),
            np.degrees(np.abs(df[ycol] - df["gt_yaw"])).mean(),
            0.5*e.mean())

def subgroup_table(df, group_col, before_cols=("pitch_corr","yaw_corr"), after_cols=("demo_pitch","demo_yaw")):
    rows=[]
    for g,dd in df.groupby(group_col, dropna=False):
        pm_b, ym_b, am_b = mae_cols(dd, before_cols[0], before_cols[1])
        pm_a, ym_a, am_a = mae_cols(dd, after_cols[0],  after_cols[1])
        rows.append({
            group_col: g,
            "N": len(dd),
            "MAE_pitch_before": pm_b, "MAE_yaw_before": ym_b, "MAE_avg_before": am_b,
            "MAE_pitch_after":  pm_a, "MAE_yaw_after":  ym_a, "MAE_avg_after":  am_a,
            "Delta_avg": am_a - am_b
        })
    T = pd.DataFrame(rows).sort_values(group_col)
    return T

def barplot_two(T, group_col, outpng, title):
    labels = T[group_col].astype(str).tolist()
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2,4.2))
    ax.bar(x - w/2, T["MAE_avg_before"], width=w, label="before")
    ax.bar(x + w/2, T["MAE_avg_after"],  width=w, label="after")
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("MAE (deg)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpng, dpi=160)
    plt.close(fig)
    print(f"[WRITE] {outpng}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv",  required=True, help="validation with gt + pitch_corr/yaw_corr")
    ap.add_argument("--test_csv", required=True, help="calibrated test (champ) with gt + pitch_corr/yaw_corr")
    ap.add_argument("--demo_csv", required=True, help="subject_id, gender, age_bin (precomputed)")
    ap.add_argument("--outdir",   required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    figsdir = os.path.join(args.outdir, "figs"); os.makedirs(figsdir, exist_ok=True)
    tabsdir = os.path.join(args.outdir, "tables"); os.makedirs(tabsdir, exist_ok=True)

    val  = load_csv(args.val_csv)
    test = load_csv(args.test_csv)
    demo = pd.read_csv(args.demo_csv)

    val  = join_demo(val, demo)
    test = join_demo(test, demo)

    val  = residuals(val)

    # --- GENDER OFFSETS ---
    if "gender" in val.columns and val["gender"].notna().any():
        Gg = group_means(val, "gender")
        print("[GENDER offsets]\n", Gg)
        test_g = apply_offsets(test, Gg, "gender", "demo")
        Tg = subgroup_table(test_g, "gender")
        tg_csv = os.path.join(tabsdir, "T_demo_gender_before_after.csv")
        Tg.to_csv(tg_csv, index=False); print(f"[WRITE] {tg_csv}")
        barplot_two(Tg, "gender", os.path.join(figsdir,"F_demo_gender_before_after.png"),
                    "MAE by gender: before vs. after demo-offset")
        # persist corrected test CSV
        out_corr = os.path.join(args.outdir, "test_CHAMPION_DEMOOFF_gender.csv")
        cols_keep = ["path","subject_id","gt_pitch","gt_yaw","pitch_corr","yaw_corr","demo_pitch","demo_yaw","gender","age_bin"]
        test_g[cols_keep].to_csv(out_corr, index=False); print(f"[WRITE] {out_corr}")
    else:
        print("[WARN] No usable gender info; skipping gender offsets")

    # --- AGE-BIN OFFSETS ---
    if "age_bin" in val.columns and val["age_bin"].notna().any():
        Ga = group_means(val, "age_bin")
        print("[AGE-BIN offsets]\n", Ga)
        test_a = apply_offsets(test, Ga, "age_bin", "demo")
        Ta = subgroup_table(test_a, "age_bin")
        ta_csv = os.path.join(tabsdir, "T_demo_age_before_after.csv")
        Ta.to_csv(ta_csv, index=False); print(f"[WRITE] {ta_csv}")
        barplot_two(Ta, "age_bin", os.path.join(figsdir,"F_demo_age_before_after.png"),
                    "MAE by age-bin: before vs. after demo-offset")
        out_corr = os.path.join(args.outdir, "test_CHAMPION_DEMOOFF_age.csv")
        cols_keep = ["path","subject_id","gt_pitch","gt_yaw","pitch_corr","yaw_corr","demo_pitch","demo_yaw","gender","age_bin"]
        test_a[cols_keep].to_csv(out_corr, index=False); print(f"[WRITE] {out_corr}")
    else:
        print("[WARN] No usable age_bin info; skipping age-bin offsets")

if __name__ == "__main__":
    main()

