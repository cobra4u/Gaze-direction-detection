import os, re, math
import pandas as pd
import numpy as np

BASE = "/users/project1/pt01281/gaze_outputs"
PRED_CSV = f"{BASE}/preds_test_gaze_exacthead_stride1.csv"
DEMO_CSV = f"{BASE}/eth_eval/tables/ETH_subject_demographics.csv"
OUT_DIR  = f"{BASE}/eth_eval/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def norm_subj(s: str) -> str:
    if pd.isna(s): return None
    s = str(s)
    # Grab things like ".../subject0001.h5" → "subject0001"
    m = re.search(r"subject\d+", s)
    return m.group(0) if m else s

def to_deg(rad):
    return rad * 180.0 / math.pi

# --- Load predictions ---
print(f"[LOAD] preds: {PRED_CSV}")
df = pd.read_csv(PRED_CSV)

# Ensure 'subject_id' present/normalized
if 'subject_id' not in df.columns:
    # Fallback: extract from 'path'
    if 'path' in df.columns:
        df['subject_id'] = df['path'].apply(norm_subj)
    else:
        raise ValueError("No 'subject_id' or 'path' in predictions CSV.")

df['subject_id'] = df['subject_id'].apply(norm_subj)

# Convert per-axis errors (deg)
# Expect gt_pitch/gt_yaw, pr_pitch/pr_yaw in radians
for c in ['gt_pitch','gt_yaw','pr_pitch','pr_yaw']:
    if c not in df.columns:
        raise ValueError(f"Predictions CSV missing column: {c}")

df['pitch_err_deg'] = (df['pr_pitch'] - df['gt_pitch']).apply(to_deg).abs()
df['yaw_err_deg']   = (df['pr_yaw']   - df['gt_yaw']).apply(to_deg).abs()
df['avg_err_deg']   = 0.5*(df['pitch_err_deg'] + df['yaw_err_deg'])

# --- Load demographics ---
print(f"[LOAD] demo : {DEMO_CSV}")
demo = pd.read_csv(DEMO_CSV)

# Normalize subject key
key_col = None
for k in ['subject_id','subject','id','subject_name']:
    if k in demo.columns:
        key_col = k
        break
if key_col is None:
    # try extract from a 'path' column if present
    if 'path' in demo.columns:
        demo['subject_id'] = demo['path'].apply(norm_subj)
        key_col = 'subject_id'
    else:
        raise ValueError("Demographics CSV has no subject-id-like column.")

demo['subject_id'] = demo[key_col].apply(norm_subj)

# Ensure gender present; if missing, set unknown (but warn)
if 'gender' not in demo.columns:
    print("[WARN] 'gender' missing in demo — filling 'unknown'")
    demo['gender'] = 'unknown'
demo['gender'] = demo['gender'].fillna('unknown').astype(str)

# Ensure age bin present; try derive from age_med if needed
if 'age_bin' not in demo.columns:
    if 'age_med' in demo.columns:
        def to_bin(a):
            if pd.isna(a): return 'unknown'
            try: a=float(a)
            except: return 'unknown'
            if a < 18: return '<18'          # unlikely in ETH test; safety
            if a <= 34: return '18-34'
            if a <= 54: return '35-54'
            return '55+'
        demo['age_bin'] = demo['age_med'].apply(to_bin)
        print("[INFO] Derived age_bin from age_med.")
    else:
        print("[WARN] 'age_bin' and 'age_med' missing — filling 'unknown'")
        demo['age_bin'] = 'unknown'

demo['age_bin'] = demo['age_bin'].fillna('unknown').astype(str)

# Disambiguate duplicates (one row per subject_id)
demo = demo.sort_values(by=['subject_id']).drop_duplicates(subset=['subject_id'], keep='first')

# --- Merge ---
mdf = df.merge(demo[['subject_id','gender','age_bin']], on='subject_id', how='left')
pre_na = mdf['gender'].isna().sum()
if pre_na>0:
    print(f"[WARN] {pre_na} rows with missing demo → will be 'unknown'")
mdf['gender']  = mdf['gender'].fillna('unknown').astype(str)
mdf['age_bin'] = mdf['age_bin'].fillna('unknown').astype(str)

# Join coverage
covered_subj = mdf[mdf['gender']!='unknown']['subject_id'].nunique()
total_subj   = mdf['subject_id'].nunique()
print(f"[COVERAGE] matched subjects: {covered_subj}/{total_subj} "
      f"({100.0*covered_subj/total_subj:.1f}%)")

# --- Aggregate tables ---
def subgroup_table(mdf, group_col, out_path):
    G = (mdf.groupby(group_col, dropna=False)
            .agg(N_samples=('avg_err_deg','size'),
                 MAE_pitch_deg=('pitch_err_deg','mean'),
                 MAE_yaw_deg  =('yaw_err_deg','mean'),
                 MAE_avg_deg  =('avg_err_deg','mean'),
                 N_subjects   =('subject_id', lambda s: pd.Series(s).nunique()))
            .reset_index())
    # nice ordering
    if group_col=='gender':
        order = ['female','male','unknown']
        G[group_col] = pd.Categorical(G[group_col], order, ordered=True)
        G = G.sort_values(group_col)
    elif group_col=='age_bin':
        order = ['18-34','35-54','55+','unknown']
        G[group_col] = pd.Categorical(G[group_col], order, ordered=True)
        G = G.sort_values(group_col)
    G.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")
    return G

Tg = subgroup_table(mdf, 'gender' , f"{OUT_DIR}/ETH_mae_by_gender.csv")
Ta = subgroup_table(mdf, 'age_bin', f"{OUT_DIR}/ETH_mae_by_agebin.csv")

# Quick head-print
print("\n[CHECK] gender table:")
print(Tg)
print("\n[CHECK] age bin table:")
print(Ta)
