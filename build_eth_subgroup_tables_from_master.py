import pandas as pd, numpy as np, os, argparse
from pathlib import Path

def to_deg(rad): return rad * 180/np.pi

def mae_deg(df):
    # average MAE over axes in degrees
    p = np.abs(to_deg(df["gt_pitch"])-to_deg(df["pr_pitch"]))
    y = np.abs(to_deg(df["gt_yaw"])-to_deg(df["pr_yaw"]))
    return pd.Series({
        "MAE_pitch_deg": p.mean(),
        "MAE_yaw_deg":   y.mean(),
        "MAE_avg_deg":   (p.mean()+y.mean())/2
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True)   # e.g. preds_test_gaze_exacthead_stride1.csv
    ap.add_argument("--demo_master_csv", required=True)  # xgaze_test_demographics_age_gender.csv
    ap.add_argument("--out_dir", required=True)     # /.../gaze_outputs/eth_eval/tables
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds_csv)
    # subject_id present in preds CSV; join on that
    dm = pd.read_csv(args.demo_master_csv)
    # normalize columns
    dm = dm.rename(columns={"gender":"gender", "age_bin":"age_bin", "subject_id":"subject_id"})
    if "age_bin" not in dm.columns and "age_mean" in dm.columns:
        # derive bins if needed
        def bin_age(a):
            try:
                a=float(a)
            except: return "unknown"
            if a<35: return "18-34"
            if a<55: return "35-54"
            return "55+"
        dm["age_bin"] = dm["age_mean"].apply(bin_age)

    # Clean gender text
    dm["gender"] = dm["gender"].astype(str).str.strip().str.title()
    dm["gender"] = dm["gender"].replace({"Male":"Men","Man":"Men","Female":"Women","Woman":"Women"})

    # Keep relevant cols
    dm = dm[["subject_id","gender","age_bin"]].drop_duplicates()

    # join
    m = df.merge(dm, on="subject_id", how="left")
    m["gender"] = m["gender"].fillna("unknown")
    m["age_bin"] = m["age_bin"].fillna("unknown")

    # Gender table
    G = m.groupby("gender", dropna=False)
    T_gender = G.apply(mae_deg).reset_index()
    T_gender["N_samples"] = G.size().values
    T_gender["N_subjects"] = G["subject_id"].nunique().values
    T_gender = T_gender.rename(columns={"gender":"group"})
    T_gender = T_gender[["group","N_samples","MAE_pitch_deg","MAE_yaw_deg","MAE_avg_deg","N_subjects"]]
    T_gender.to_csv(out_dir/"ETH_mae_by_gender.csv", index=False)

    # Age table
    A = m.groupby("age_bin", dropna=False)
    T_age = A.apply(mae_deg).reset_index()
    T_age["N_samples"] = A.size().values
    T_age["N_subjects"] = A["subject_id"].nunique().values
    T_age = T_age.rename(columns={"age_bin":"group"})
    T_age = T_age[["group","N_samples","MAE_pitch_deg","MAE_yaw_deg","MAE_avg_deg","N_subjects"]]
    T_age.to_csv(out_dir/"ETH_mae_by_agebin.csv", index=False)

    # Also rewrite T5_3 (combined) for convenience
    T53_g = T_gender.copy(); T53_g.insert(0,"group_type","gender")
    T53_a = T_age.copy();    T53_a.insert(0,"group_type","age_bin")
    T53 = pd.concat([T53_g, T53_a], ignore_index=True)
    T53.to_csv(out_dir/"T5_3_subgroup_mae.csv", index=False)

    print("[WRITE]", out_dir/"ETH_mae_by_gender.csv")
    print("[WRITE]", out_dir/"ETH_mae_by_agebin.csv")
    print("[WRITE]", out_dir/"T5_3_subgroup_mae.csv")

if __name__ == "__main__":
    main()
