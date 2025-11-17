import os, re, argparse
import numpy as np, pandas as pd

TO_DEG = 180/np.pi

def wrap180(a): return ((a+180.0)%360.0)-180.0
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def extract_sid_from_path(s):
    if not isinstance(s, str): return None
    m = re.search(r"(subject\\d{4})", s)
    return m.group(1) if m else None

def choose_pred_cols(df):
    if {"pitch_corr","yaw_corr"}.issubset(df.columns):
        return "pitch_corr","yaw_corr"
    if {"pr_pitch_calib","pr_yaw_calib"}.issubset(df.columns):
        return "pr_pitch_calib","pr_yaw_calib"
    return "pr_pitch","pr_yaw"

def load_preds(csv_path):
    df = pd.read_csv(csv_path)
    # sanity: need gt + some prediction cols
    have_pred = any(s.issubset(df.columns) for s in [
        {"pitch_corr","yaw_corr"},
        {"pr_pitch_calib","pr_yaw_calib"},
        {"pr_pitch","pr_yaw"},
    ])
    if not {"gt_pitch","gt_yaw"}.issubset(df.columns) or not have_pred:
        raise ValueError(f"[{csv_path}] missing required columns")
    if "subject_id" not in df.columns:
        df["subject_id"] = df.get("path","").astype(str).map(extract_sid_from_path).fillna("unknown")
    return df

def axis_mae_deg(df, pred_cols=None):
    gp = df["gt_pitch"].astype(float)*TO_DEG
    gy = df["gt_yaw"].astype(float)*TO_DEG
    if pred_cols is None: pred_cols = choose_pred_cols(df)
    pp = df[pred_cols[0]].astype(float)*TO_DEG
    py = df[pred_cols[1]].astype(float)*TO_DEG
    ep = wrap180(pp - gp).abs()
    ey = wrap180(py - gy).abs()
    ea = (ep + ey)/2.0
    return float(ep.mean()), float(ey.mean()), float(ea.mean())

def withinK_deg(df, K_list=(5,10,15,20), pred_cols=None):
    gp = df["gt_pitch"].astype(float)*TO_DEG
    gy = df["gt_yaw"].astype(float)*TO_DEG
    if pred_cols is None: pred_cols = choose_pred_cols(df)
    pp = df[pred_cols[0]].astype(float)*TO_DEG
    py = df[pred_cols[1]].astype(float)*TO_DEG
    ep = wrap180(pp - gp).abs()
    ey = wrap180(py - gy).abs()
    ea = (ep + ey)/2.0
    rows = []
    for K in K_list:
        rows.append({
            "K_deg": K,
            "within_pitch&yaw_%": 100.0 * float(((ep<=K) & (ey<=K)).mean()),
            "within_avg_%":       100.0 * float((ea<=K).mean())
        })
    return pd.DataFrame(rows)

def load_demo_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["subject_id","gender","age_bin"])
    d = pd.read_csv(path)
    # normalize columns
    if "subject_id" not in d.columns:
        for k in ("subject","sid"):
            if k in d.columns: d = d.rename(columns={k:"subject_id"})
    if "gender" not in d.columns and "sex" in d.columns:
        d = d.rename(columns={"sex":"gender"})
    if "age_bin" not in d.columns:
        for k in ("age_group","agebin","ageBin"):
            if k in d.columns: d = d.rename(columns={k:"age_bin"})
    for c in ("subject_id","gender","age_bin"):
        if c not in d.columns:
            return pd.DataFrame(columns=["subject_id","gender","age_bin"])
    d["subject_id"] = d["subject_id"].astype(str)
    d["gender"]  = d["gender"].fillna("unknown").astype(str)
    d["age_bin"] = d["age_bin"].fillna("unknown").astype(str)
    return d[["subject_id","gender","age_bin"]].drop_duplicates()

def subgroup_table_two_runs(df_raw, df_champ, demo_df):
    # subject ids
    for df in (df_raw, df_champ):
        if "subject_id" not in df.columns:
            df["subject_id"] = df.get("path","").astype(str).map(extract_sid_from_path).fillna("unknown")
    if demo_df is None or demo_df.empty:
        demo_df = pd.DataFrame(columns=["subject_id","gender","age_bin"])

    def compute_block(df):
        pred_cols = choose_pred_cols(df)
        gp = df["gt_pitch"].astype(float)*TO_DEG
        gy = df["gt_yaw"].astype(float)*TO_DEG
        pp = df[pred_cols[0]].astype(float)*TO_DEG
        py = df[pred_cols[1]].astype(float)*TO_DEG
        ep = wrap180(pp - gp).abs()
        ey = wrap180(py - gy).abs()
        ea = (ep + ey)/2.0
        out = df.copy()
        out["err_p"] = ep; out["err_y"] = ey; out["err_a"] = ea
        out["within10_avg"] = (ea <= 10.0).astype(int)
        return out

    R = compute_block(df_raw.merge(demo_df, on="subject_id", how="left"))
    C = compute_block(df_champ.merge(demo_df, on="subject_id", how="left"))
    for df in (R, C):
        df["gender"]  = df["gender"].fillna("unknown")
        df["age_bin"] = df["age_bin"].fillna("unknown")

    def agg_one(df, by_col, run_name):
        A = (df.groupby(by_col, dropna=False)
                .agg(N_samples=("err_a","size"),
                     MAE_pitch_deg=("err_p","mean"),
                     MAE_yaw_deg=("err_y","mean"),
                     MAE_avg_deg=("err_a","mean"),
                     within10_avg_pct=("within10_avg","mean"))
                .reset_index())
        A["within10_avg_pct"] = 100.0 * A["within10_avg_pct"].astype(float)
        subs = df[[by_col,"subject_id"]].dropna().drop_duplicates()
        S = subs.groupby(by_col, dropna=False).size().reset_index(name="N_subjects")
        A = A.merge(S, on=by_col, how="left")
        A.insert(0,"group_type", by_col)
        A.rename(columns={by_col:"group"}, inplace=True)
        A.insert(1,"run", run_name)
        return A

    G_raw = agg_one(R, "gender",  "raw")
    G_ch  = agg_one(C, "gender",  "champion")
    A_raw = agg_one(R, "age_bin", "raw")
    A_ch  = agg_one(C, "age_bin", "champion")

    T = pd.concat([G_raw,G_ch,A_raw,A_ch], ignore_index=True)

    # ---- FIXED SORT: use helper sort keys, then drop them ----
    map_gt = {"gender":0, "age_bin":1}
    map_run = {"raw":0, "champion":1}
    T["o_gt"]  = T["group_type"].map(map_gt).fillna(9).astype(int)
    T["o_g"]   = T["group"].astype(str)
    T["o_run"] = T["run"].map(map_run).fillna(9).astype(int)
    T = T.sort_values(["o_gt","o_g","o_run"]).drop(columns=["o_gt","o_g","o_run"])

    cols = ["group_type","group","run","N_samples","N_subjects",
            "MAE_pitch_deg","MAE_yaw_deg","MAE_avg_deg","within10_avg_pct"]
    return T[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True)
    ap.add_argument("--champ_csv",    required=True)
    ap.add_argument("--outdir",       required=True)
    ap.add_argument("--demo_csv",     default="/users/project1/pt01281/gaze_outputs/eth_eval/tables/ETH_subject_demographics.csv")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    raw_df   = load_preds(args.baseline_csv)
    champ_df = load_preds(args.champ_csv)

    # Table 5.4 — calibration stages (if files exist)
    stages = [
        ("raw_baseline", args.baseline_csv),
        ("piecewise", "/users/project1/pt01281/gaze_outputs/eth_eval_calib/preds_test_CALIB_piecewise.csv"),
        ("vec_affine", "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_vec_affine.csv"),
        ("poly2d", "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_poly2d.csv"),
        ("blend", "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND.csv"),
        ("blend_offgbin2", "/users/project1/pt01281/gaze_outputs/eth_eval_calib/test_preds_BLEND_OFFGBIN2.csv"),
        ("champion", args.champ_csv),
    ]
    recs = []
    mae_p_raw, mae_y_raw, mae_a_raw = axis_mae_deg(raw_df)
    for name, path in stages:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: not found -> {path}")
            continue
        df = raw_df if name=="raw_baseline" else load_preds(path)
        mp, my, ma = axis_mae_deg(df)
        recs.append({
            "stage": name,
            "MAE_pitch_deg": mp,
            "MAE_yaw_deg":   my,
            "MAE_avg_deg":   ma,
            "Delta_vs_raw_avg_deg": ma - mae_a_raw
        })
    T54 = pd.DataFrame(recs)
    T54.to_csv(os.path.join(args.outdir, "T5_4_calibration_stages.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.outdir, 'T5_4_calibration_stages.csv')}")

    # Table 5.5 — within-K raw vs champion
    W_raw = withinK_deg(raw_df);  W_raw.insert(0,"run","raw")
    W_ch  = withinK_deg(champ_df);W_ch.insert(0,"run","champion")
    T55 = pd.concat([W_raw,W_ch], ignore_index=True)
    T55.to_csv(os.path.join(args.outdir, "T5_5_withinK_raw_vs_champ.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.outdir, 'T5_5_withinK_raw_vs_champ.csv')}")

    # Table 5.6 — subgroup MAE & within-10° raw vs champion
    demo = load_demo_csv(args.demo_csv)
    T56 = subgroup_table_two_runs(raw_df, champ_df, demo)
    T56.to_csv(os.path.join(args.outdir, "T5_6_subgroup_raw_vs_champ.csv"), index=False)
    print(f"[WRITE] {os.path.join(args.outdir, 'T5_6_subgroup_raw_vs_champ.csv')}")

if __name__ == "__main__":
    main()
