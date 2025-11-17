#!/usr/bin/env python3
# Location: /users/kdm/divjots2002/gazemain/scan_signs_calib.py
import argparse, numpy as np, pandas as pd

def pick_cols(df):
    cols = df.columns
    pp = 'pred_pitch' if 'pred_pitch' in cols else ('pr_pitch' if 'pr_pitch' in cols else None)
    py = 'pred_yaw'   if 'pred_yaw'   in cols else ('pr_yaw'   if 'pr_yaw'   in cols else None)
    if pp is None or py is None: raise ValueError("Need pred_pitch/pred_yaw or pr_pitch/pr_yaw")
    if 'gt_pitch' not in cols or 'gt_yaw' not in cols:
        raise ValueError("Need gt_pitch and gt_yaw in VAL file")
    return pp, py, 'gt_pitch', 'gt_yaw'

def fit_eval(df, pp, py, gp, gy, s_p=1, s_y=1):
    X = df[[pp, py]].to_numpy().copy()
    X[:,0] *= s_p
    X[:,1] *= s_y
    Y = df[[gp, gy]].to_numpy()
    X1 = np.concatenate([X, np.ones((len(X),1))], axis=1)
    W, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    A, b = W[:2,:], W[2,:]
    P = (X @ A) + b
    e = np.abs(P - Y)
    mae_p = e[:,0].mean(); mae_y = e[:,1].mean()
    return mae_p, mae_y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="VAL CSV with GT (rad)")
    args = ap.parse_args()
    df = pd.read_csv(args.val)
    pp, py, gp, gy = pick_cols(df)
    variants = {
        "(+pitch, +yaw)" : ( 1,  1),
        "(-pitch, +yaw)" : (-1,  1),
        "(+pitch, -yaw)" : ( 1, -1),
        "(-pitch, -yaw)" : (-1, -1),
    }
    best = None
    for name,(sp,sy) in variants.items():
        mp, my = fit_eval(df, pp, py, gp, gy, sp, sy)
        avg = 0.5*(mp+my)
        print(f"{name:16s}  MAE(rad): p={mp:.6f} y={my:.6f} avg={avg:.6f} | deg avg={avg*180/np.pi:.3f}")
        if best is None or avg < best[0]:
            best = (avg, name, sp, sy)
    print("\nBest:", best[1], "| avg deg =", best[0]*180/np.pi)
    print("Use flags: ", "--flip-pitch" if best[2]<0 else "", "--flip-yaw" if best[3]<0 else "")
if __name__ == "__main__":
    main()

