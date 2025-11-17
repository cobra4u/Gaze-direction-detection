#!/usr/bin/env python3
# Location: /users/kdm/divjots2002/gazemain/calibrate_isotonic_ethxgaze.py
"""
Fast per-axis isotonic calibration (Pool-Adjacent-Violators) for gaze angles (radians).

- Fits two 1D monotone functions on VAL: f_p(pred_pitch)->gt_pitch, f_y(pred_yaw)->gt_yaw
- Applies to TEST to produce pitch_corr, yaw_corr
- Auto-detects pr_* vs pred_* columns
- CPU-friendly (O(N log N)), no deps beyond numpy/pandas

Usage:
  python -u /users/kdm/divjots2002/gazemain/calibrate_isotonic_ethxgaze.py \
    --val  /users/project1/pt01281/gaze_outputs/eth_eval_calib/preds_test_gaze_exacthead_stride1_CALIB_10p.csv \
    --test /users/kdm/divjots2002/gazemain/my_gaze_outputs/preds_eth_test.csv \
    --out  /users/project1/pt01281/gaze_outputs/calibration/test_preds_iso.csv
"""

import argparse, numpy as np, pandas as pd

def pick_cols(df):
    cols = df.columns
    pp = 'pred_pitch' if 'pred_pitch' in cols else ('pr_pitch' if 'pr_pitch' in cols else None)
    py = 'pred_yaw'   if 'pred_yaw'   in cols else ('pr_yaw'   if 'pr_yaw'   in cols else None)
    if pp is None or py is None:
        raise ValueError(f"Need pred_pitch/pred_yaw or pr_pitch/pr_yaw; have: {list(cols)}")
    gp = 'gt_pitch' if 'gt_pitch' in cols else None
    gy = 'gt_yaw'   if 'gt_yaw'   in cols else None
    return pp, py, gp, gy

def pav_isotonic(x, y, increasing=True):
    """
    Pool-Adjacent-Violators for 1D monotonic regression.
    Returns a function f(new_x) via piecewise-linear interpolation on unique x.
    """
    # sort by x
    idx = np.argsort(x)
    x = x[idx].astype(float)
    y = y[idx].astype(float)

    # if decreasing desired, flip sign on y during fit then flip back in f
    sign = 1.0 if increasing else -1.0
    y_work = sign * y

    # PAV: maintain blocks with weighted means
    w = np.ones_like(y_work)
    v = y_work.copy()
    n = len(v)
    # block starts
    start = np.arange(n)
    for i in range(n-1):
        if v[i] > v[i+1] + 1e-12:  # violation
            j = i
            while j >= 0 and v[j] > v[j+1] + 1e-12:
                # merge blocks j and j+1
                wnew = w[j] + w[j+1]
                vnew = (w[j]*v[j] + w[j+1]*v[j+1]) / wnew
                w[j] = wnew; v[j] = vnew
                # shift left the arrays from j+1
                w = np.delete(w, j+1)
                v = np.delete(v, j+1)
                x = np.delete(x, j+1)
                # continue checking backward
                j -= 1

    # Now v is nondecreasing (in y_work). Build isotonic (x_unique, y_iso)
    xu = x
    yu = sign * v  # flip back

    # unique x guards (already aggregated by merges)
    # Build piecewise-linear interpolation
    def f(new_x):
        new_x = np.asarray(new_x, dtype=float)
        # clamp outside range
        yout = np.empty_like(new_x)
        lo, hi = xu[0], xu[-1]
        ylo, yhi = yu[0], yu[-1]
        left = new_x <= lo
        right = new_x >= hi
        mid = (~left) & (~right)
        yout[left] = ylo
        yout[right] = yhi
        if np.any(mid):
            xm = new_x[mid]
            # find indices for linear segments
            # np.searchsorted gives insertion point; subtract 1 for left index
            j = np.searchsorted(xu, xm) - 1
            j = np.clip(j, 0, len(xu)-2)
            x0, x1 = xu[j], xu[j+1]
            y0, y1 = yu[j], yu[j+1]
            t = (xm - x0) / (x1 - x0 + 1e-12)
            yout[mid] = y0 + t*(y1 - y0)
        return yout

    return f

def mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    val  = pd.read_csv(args.val)
    test = pd.read_csv(args.test)

    pp, py, gp, gy = pick_cols(val)
    if gp is None or gy is None:
        raise ValueError("VAL must include gt_pitch and gt_yaw (radians).")

    # to numpy
    p_pred = pd.to_numeric(val[pp], errors='coerce').to_numpy()
    y_pred = pd.to_numeric(val[py], errors='coerce').to_numpy()
    p_gt   = pd.to_numeric(val[gp], errors='coerce').to_numpy()
    y_gt   = pd.to_numeric(val[gy], errors='coerce').to_numpy()

    m = ~np.isnan(p_pred) & ~np.isnan(y_pred) & ~np.isnan(p_gt) & ~np.isnan(y_gt)
    p_pred, y_pred, p_gt, y_gt = p_pred[m], y_pred[m], p_gt[m], y_gt[m]

    # Fit isotonic per-axis (increasing is usually correct given positive corr)
    f_pitch = pav_isotonic(p_pred, p_gt, increasing=True)
    f_yaw   = pav_isotonic(y_pred, y_gt, increasing=True)

    # Report VAL MAE after isotonic (per-axis)
    p_corr_val = f_pitch(p_pred)
    y_corr_val = f_yaw(y_pred)
    vm = 0.5*(mae(p_corr_val, p_gt) + mae(y_corr_val, y_gt))
    to_deg = 180/np.pi
    print(f"[VAL iso] MAE(rad): avg={vm:.6f} | deg avg={vm*to_deg:.3f}")

    # Apply to TEST
    pp_t, py_t, _, _ = pick_cols(test)
    tp = pd.to_numeric(test[pp_t], errors='coerce').to_numpy()
    ty = pd.to_numeric(test[py_t], errors='coerce').to_numpy()
    test["pitch_corr"] = f_pitch(tp)
    test["yaw_corr"]   = f_yaw(ty)
    test.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out}")

if __name__ == "__main__":
    main()

