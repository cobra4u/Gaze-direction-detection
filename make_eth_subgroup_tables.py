import os, argparse, re
import numpy as np
import pandas as pd

TO_DEG = 180.0/np.pi

def wrap180(a_deg):
    return ((a_deg + 180.0) % 360.0) - 180.0

def load_preds(path):
    df = pd.read_csv(path)
    # Ensure subject_id
    if "subject_id" not in df.columns:
        # Try to extract subject???? from path like .../subject0001.h5::...
        sid = df["path"].astype(str).str.extract(r"(subject\d{4})", expand=False)
        df["subject_id"] = sid.fillna("unknown")
    # Required columns check
    req = ["gt_pitch","gt_yaw","pr_pitch","pr_yaw","subject_id"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Preds CSV missing columns: {miss}")
    # Errors in degrees with seam-safe wrapping
    dth = wrap180((df["pr_pitch"].astype(float) - df["gt_pitch"].astype(float)) * TO_DEG)
    dps = wrap180((df["pr_yaw"].astype(float)   - df["gt_yaw"].astype(float))   * TO_DEG)
    df["err_pitch_deg"] = dth
    df["err_yaw_deg"]   = dps
    df["err_avg_deg"]   = (dth.abs() + dps.abs())/2.0
    return df

def agg_table(df, key, out_csv):
    g = (df
         .groupby(key, dropna=False)
         .agg(N_samples = ("err_avg_deg","size"),
              MAE_pitch_deg = ("err_pitch_deg", lambda s: float(s.abs().mean())),
              MAE_yaw_deg   = ("err_yaw_deg",   lambda s: float(s.abs().mean())),
              MAE_avg_deg   = ("err_avg_deg",   "mean"))
         .reset_index()
        )

    # Also include subject counts per group if possible
    if "subject_id" in df.columns:
        subs = (df[["subject_id", key]]
                .drop_duplicates()
                .groupby(key, dropna=False)
                .size()
                .rename("N_subjects")
                .reset_index())
        g = g.merge(subs, on=key, how="left")

    # Order groups nicely if key is categorical-ish
    if key == "gender":
        order = ["female","male","unknown"]
        g[key] = pd.Categorical(g[key], categories=order, ordered=True)
        g = g.sort_values(key)
    elif key == "age_bin":
        order = ["18-34","35-54","55+","unknown"]
        g[key] = pd.Categorical(g[key], categories=order, ordered=True)
        g = g.sort_values(key)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    g.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True)
    ap.add_argument("--demo_csv",  required=True)
    ap.add_argument("--out_dir",   required=True)
    args = ap.parse_args()

    print("[LOAD] preds:", args.preds_csv)
    preds = load_preds(args.preds_csv)

    print("[LOAD] demo:", args.demo_csv)
    demo = pd.read_csv(args.demo_csv)
    if not {"subject_id","gender","age_bin"}.issubset(demo.columns):
        raise ValueError("Demographics CSV must contain subject_id, gender, age_bin")

    df = preds.merge(demo[["subject_id","gender","age_bin"]], on="subject_id", how="left")
    df["gender"]  = df["gender"].fillna("unknown")
    df["age_bin"] = df["age_bin"].fillna("unknown")

    # Write tables
    out_gender = os.path.join(args.out_dir, "ETH_mae_by_gender.csv")
    out_age    = os.path.join(args.out_dir, "ETH_mae_by_agebin.csv")

    agg_table(df, "gender",  out_gender)
    agg_table(df, "age_bin", out_age)

if __name__ == "__main__":
    main()
