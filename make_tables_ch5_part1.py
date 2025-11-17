import os, re, json, argparse
import numpy as np, pandas as pd

TO_DEG = 180/np.pi
def wrap180(a): return ((a+180.0)%360.0)-180.0

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def extract_sid_from_path(s):
    if not isinstance(s, str): return None
    m = re.search(r"(subject\d{4})", s)
    return m.group(1) if m else None

def load_preds(csv_path):
    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        df["subject_id"] = df.get("path","").astype(str).map(extract_sid_from_path).fillna("unknown")
    need = {"gt_pitch","gt_yaw","pr_pitch","pr_yaw"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"[{os.path.basename(csv_path)}] missing columns: {miss}")
    return df

def load_metrics_json(p):
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_withinK_csv(p):
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

def compute_axis_mae(df):
    gp = df["gt_pitch"].astype(float); gy = df["gt_yaw"].astype(float)
    pp = df["pr_pitch"].astype(float); py = df["pr_yaw"].astype(float)
    ep = wrap180((pp - gp) * TO_DEG).abs()
    ey = wrap180((py - gy) * TO_DEG).abs()
    ea = (ep + ey)/2.0
    return float(ep.mean()), float(ey.mean()), float(ea.mean())

def bin_edges_0_90(step=15):
    return np.arange(0, 90+1, step, dtype=float)

def per_pose_bin_table(df):
    gp = df["gt_pitch"].astype(float)*TO_DEG
    gy = df["gt_yaw"].astype(float)*TO_DEG
    pp = df["pr_pitch"].astype(float)*TO_DEG
    py = df["pr_yaw"].astype(float)*TO_DEG

    ep = wrap180(pp - gp).abs()
    ey = wrap180(py - gy).abs()
    ea = (ep + ey)/2.0

    yaw_abs   = gy.abs()
    pitch_abs = gp.abs()

    be = bin_edges_0_90(15)
    yaw_bin   = pd.cut(yaw_abs,   bins=be, include_lowest=True, right=True)
    pitch_bin = pd.cut(pitch_abs, bins=be, include_lowest=True, right=True)

    T = (pd.DataFrame({
            "yaw_bin": yaw_bin.astype(str),
            "pitch_bin": pitch_bin.astype(str),
            "err_p": ep, "err_y": ey, "err_a": ea
        })
        .groupby(["pitch_bin","yaw_bin"], dropna=False)
        .agg(N=("err_a","size"),
             MAE_pitch_deg=("err_p","mean"),
             MAE_yaw_deg=("err_y","mean"),
             MAE_avg_deg=("err_a","mean"))
        .reset_index()
    )

    def sort_key(bin_str):
        m = re.findall(r"[-+]?\d+\.?\d*", bin_str)
        if len(m)>=2:
            return (float(m[0]), float(m[1]))
        return (9999, 9999)

    T["yaw_sort"]   = T["yaw_bin"].map(sort_key)
    T["pitch_sort"] = T["pitch_bin"].map(sort_key)
    T = T.sort_values(["pitch_sort","yaw_sort"]).drop(columns=["yaw_sort","pitch_sort"])
    return T

def load_demo_csv(path):
    if not path or not os.path.exists(path): return None
    d = pd.read_csv(path)
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
            print(f"[WARN] demographics csv missing {c}; subgroup table will be 'unknown' only.")
            return pd.DataFrame(columns=["subject_id","gender","age_bin"])
    d["gender"]  = d["gender"].fillna("unknown").astype(str)
    d["age_bin"] = d["age_bin"].fillna("unknown").astype(str)
    return d[["subject_id","gender","age_bin"]].drop_duplicates()

def subgroup_table(df_preds, demo_df):
    if demo_df is None:
        demo_df = pd.DataFrame(columns=["subject_id","gender","age_bin"])

    df = df_preds.merge(demo_df, on="subject_id", how="left")
    df["gender"]  = df["gender"].fillna("unknown")
    df["age_bin"] = df["age_bin"].fillna("unknown")

    gp = df["gt_pitch"].astype(float)*TO_DEG
    gy = df["gt_yaw"].astype(float)*TO_DEG
    pp = df["pr_pitch"].astype(float)*TO_DEG
    py = df["pr_yaw"].astype(float)*TO_DEG

    ep = wrap180(pp - gp).abs()
    ey = wrap180(py - gy).abs()
    ea = (ep + ey)/2.0
    df["err_p"] = ep; df["err_y"] = ey; df["err_a"] = ea

    # --- gender aggregates ---
    G_base = (df.groupby("gender", dropna=False)
                .agg(N_samples=("err_a","size"),
                     MAE_pitch_deg=("err_p","mean"),
                     MAE_yaw_deg=("err_y","mean"),
                     MAE_avg_deg=("err_a","mean"))
                .reset_index())
    G_subj = (df[["gender","subject_id"]]
                .dropna()
                .drop_duplicates()
                .groupby("gender", dropna=False)
                .size().reset_index(name="N_subjects"))
    G = G_base.merge(G_subj, on="gender", how="left")
    G.insert(0, "group_type", "gender")
    G.rename(columns={"gender":"group"}, inplace=True)

    # --- age-bin aggregates ---
    A_base = (df.groupby("age_bin", dropna=False)
                .agg(N_samples=("err_a","size"),
                     MAE_pitch_deg=("err_p","mean"),
                     MAE_yaw_deg=("err_y","mean"),
                     MAE_avg_deg=("err_a","mean"))
                .reset_index())
    A_subj = (df[["age_bin","subject_id"]]
                .dropna()
                .drop_duplicates()
                .groupby("age_bin", dropna=False)
                .size().reset_index(name="N_subjects"))
    A = A_base.merge(A_subj, on="age_bin", how="left")
    A.insert(0, "group_type", "age_bin")
    A.rename(columns={"age_bin":"group"}, inplace=True)

    T = pd.concat([G,A], ignore_index=True)
    # keep a stable order: gender first then age bins
    order = (T["group_type"].map({"gender":0,"age_bin":1})
             .fillna(9).astype(int))
    T["_ord"] = order
    T = T.sort_values(["_ord","group"]).drop(columns=["_ord"])
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_csv", required=True,
                    help="Baseline predictions CSV (raw)")
    ap.add_argument("--metrics_json", default="/users/project1/pt01281/gaze_outputs/eth_eval/eth_metrics.json")
    ap.add_argument("--withink_csv", default="/users/project1/pt01281/gaze_outputs/eth_eval/eth_withinK.csv")
    ap.add_argument("--demo_csv", default="/users/project1/pt01281/gaze_outputs/eth_eval/tables/ETH_subject_demographics.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # ---- Table 5.1 ----
    M = load_metrics_json(args.metrics_json)
    W = load_withinK_csv(args.withink_csv)
    if M is None:
        print("[INFO] metrics json missing; recomputing from baseline CSV for Table 5.1")
        dfm = load_preds(args.baseline_csv)
        mae_p, mae_y, mae_a = compute_axis_mae(dfm)
        M = {"N": int(len(dfm)),
             "MAE_pitch_deg": mae_p,
             "MAE_yaw_deg": mae_y,
             "MAE_avg_deg": mae_a}
    rows = [
        {"Metric":"Average MAE (deg)", "Value": M["MAE_avg_deg"]},
        {"Metric":"Pitch MAE (deg)",   "Value": M["MAE_pitch_deg"]},
        {"Metric":"Yaw MAE (deg)",     "Value": M["MAE_yaw_deg"]},
    ]
    if W is not None and set(["K_deg","within_pitch&yaw_%","within_avg_%"]).issubset(W.columns):
        for _,r in W.iterrows():
            rows.append({"Metric": f"Within {int(r['K_deg'])}° (avg)",        "Value": r["within_avg_%"]})
            rows.append({"Metric": f"Within {int(r['K_deg'])}° (pitch&yaw)",  "Value": r["within_pitch&yaw_%"]})
    else:
        rows.append({"Metric":"Within 5°/10°/15°/20°", "Value":"— (file missing)"})
    T51 = pd.DataFrame(rows)
    out_51 = os.path.join(args.outdir, "T5_1_eth_baseline_metrics.csv")
    T51.to_csv(out_51, index=False)
    print(f"[WRITE] {out_51}")

    # ---- Table 5.2 ----
    dfb = load_preds(args.baseline_csv)
    T52 = per_pose_bin_table(dfb)
    out_52 = os.path.join(args.outdir, "T5_2_per_pose_mae.csv")
    T52.to_csv(out_52, index=False)
    print(f"[WRITE] {out_52}")

    # ---- Table 5.3 ----
    demo = load_demo_csv(args.demo_csv)
    T53 = subgroup_table(dfb, demo)
    out_53 = os.path.join(args.outdir, "T5_3_subgroup_mae.csv")
    T53.to_csv(out_53, index=False)
    print(f"[WRITE] {out_53}")

if __name__ == "__main__":
    main()
