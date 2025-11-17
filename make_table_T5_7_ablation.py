import os, glob, json
import numpy as np
import pandas as pd

TO_DEG = 180.0/np.pi

def load_csv(path):
    df = pd.read_csv(path)
    # normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df

def pick_pred_cols(df):
    # Prefer corrected columns if present
    if {'pitch_corr','yaw_corr'}.issubset(df.columns):
        return 'pitch_corr','yaw_corr'
    # Else use calibrated heads if present
    if {'pr_pitch_calib','pr_yaw_calib'}.issubset(df.columns):
        return 'pr_pitch_calib','pr_yaw_calib'
    # Fallback to raw predictions
    return 'pr_pitch','pr_yaw'

def compute_metrics(df):
    # Ground-truth must be radians (ETH/XGaze convention)
    gt_p, gt_y = 'gt_pitch','gt_yaw'
    pr_p, pr_y = pick_pred_cols(df)

    if not {gt_p,gt_y,pr_p,pr_y}.issubset(df.columns):
        raise ValueError(f"Missing required columns in dataframe: have {df.columns.tolist()}")

    # absolute axiswise error (deg)
    de_p = np.abs((df[pr_p].astype(float) - df[gt_p].astype(float)) * TO_DEG)
    de_y = np.abs((df[pr_y].astype(float) - df[gt_y].astype(float)) * TO_DEG)
    de_avg = 0.5*(de_p + de_y)

    out = {
        'N_samples': int(len(df)),
        'MAE_pitch_deg': float(de_p.mean()),
        'MAE_yaw_deg': float(de_y.mean()),
        'MAE_avg_deg': float(de_avg.mean()),
        'Within10_avg_%': float((de_avg<=10.0).mean()*100.0),
    }
    return out

def stage_label_from_filename(fname):
    name = os.path.basename(fname).lower()
    # Make readable labels; add/extend rules as needed
    if name == 'test_champion.csv':
        return 'champion (calibrated)'
    if 'champ_blend' in name or (name=='test_champ_blend.csv'):
        return 'blend (calibrated)'
    if 'subcenter' in name or 'sub_center' in name:
        return 'sub-centering (calibrated)'
    if 'blend_offgbin' in name and 'grid2d' in name:
        return 'blend+offsets+grid2D'
    if 'blend_offgbin' in name:
        return 'blend+offsets'
    if 'pw_offgbin' in name:
        return 'piecewise+offsets'
    if 'blend' in name:
        return 'blend (other)'
    if 'poly' in name:
        return 'affine/polynomial'
    if 'piecewise' in name:
        return 'piecewise'
    if 'isosoft' in name or 'iso' in name:
        return 'isotonic (soft)'
    return name

def main():
    base_dir = "/users/project1/pt01281/gaze_outputs"
    calib_dir = os.path.join(base_dir, "eth_eval_calib")
    out_dir = os.path.join(base_dir, "eth_eval", "tables")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline (uncalibrated)
    baseline_csv = os.path.join(base_dir, "preds_test_gaze_exacthead_stride1.csv")
    if not os.path.exists(baseline_csv):
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")

    # Candidate stages (only those that exist will be used)
    candidates = [
        os.path.join(calib_dir, "test_CHAMPION.csv"),
        os.path.join(calib_dir, "test_CHAMP_BLEND.csv"),
        os.path.join(calib_dir, "test_CHAMP_SUBCENTER.csv"),
        os.path.join(calib_dir, "test_preds_BLEND.csv"),
        os.path.join(calib_dir, "test_preds_BLEND_OFFGBIN2.csv"),
        os.path.join(calib_dir, "test_preds_PW_OFFGBIN2.csv"),
        os.path.join(calib_dir, "test_preds_BLEND_OFFGBIN2_GRID2D.csv"),
        os.path.join(calib_dir, "test_preds_BLEND_OFFGBIN2_GRID2D_SOFT.csv"),
        os.path.join(calib_dir, "test_preds_BLEND_OFFGBIN2_ISOsoft.csv"),
        # add any others you've produced:
        # os.path.join(calib_dir, "test_TTA_ISOsoft_b0.7.csv"),  # (careful if those were in degrees)
    ]
    candidates = [p for p in candidates if os.path.exists(p)]

    rows = []

    # Baseline first
    df_base = load_csv(baseline_csv)
    m_base = compute_metrics(df_base)
    rows.append({
        'Stage': 'baseline (raw)',
        **m_base
    })

    # Stages
    for f in candidates:
        try:
            df = load_csv(f)
            m = compute_metrics(df)
            # delta vs baseline
            m['Delta_avg_deg_vs_baseline'] = m['MAE_avg_deg'] - m_base['MAE_avg_deg']
            m['CSV'] = f
            rows.append({
                'Stage': stage_label_from_filename(f),
                **m
            })
        except Exception as e:
            # Keep going; note the failure
            rows.append({
                'Stage': stage_label_from_filename(f),
                'N_samples': 0,
                'MAE_pitch_deg': np.nan,
                'MAE_yaw_deg': np.nan,
                'MAE_avg_deg': np.nan,
                'Within10_avg_%': np.nan,
                'Delta_avg_deg_vs_baseline': np.nan,
                'CSV': f,
                'Note': f'ERROR: {e}'
            })

    T = pd.DataFrame(rows)

    # Sort by MAE if available, keeping baseline first
    def sort_key(row):
        if row['Stage'].startswith('baseline'):
            return (-1, 9e9)
        return (0, row['MAE_avg_deg'] if pd.notnull(row['MAE_avg_deg']) else 9e9)
    T = pd.DataFrame(sorted(T.to_dict('records'), key=sort_key))

    # Round nicely
    for col in ['MAE_pitch_deg','MAE_yaw_deg','MAE_avg_deg','Delta_avg_deg_vs_baseline','Within10_avg_%']:
        if col in T.columns:
            T[col] = T[col].astype(float).round(2)

    out_csv = os.path.join(out_dir, "T5_7_ablation_bars.csv")
    T.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")

if __name__ == "__main__":
    main()
