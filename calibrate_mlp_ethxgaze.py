#!/usr/bin/env python3
# Location: /users/kdm/divjots2002/gazemain/calibrate_mlp_ethxgaze.py
import argparse, os, numpy as np, pandas as pd

# Pure-numpy 2-layer MLP with ReLU, trained by LBFGS-like scipy is overkill; we'll do simple Adam-ish GD.
def pick_cols(df):
    cols = df.columns
    pp = 'pred_pitch' if 'pred_pitch' in cols else ('pr_pitch' if 'pr_pitch' in cols else None)
    py = 'pred_yaw'   if 'pred_yaw'   in cols else ('pr_yaw'   if 'pr_yaw'   in cols else None)
    if pp is None or py is None:
        raise ValueError("Need pred_pitch/pred_yaw or pr_pitch/pr_yaw")
    if 'gt_pitch' not in cols or 'gt_yaw' not in cols:
        raise ValueError("VAL must include gt_pitch and gt_yaw")
    return pp, py, 'gt_pitch','gt_yaw'

def relu(x): return np.maximum(0, x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rs = np.random.RandomState(args.seed)

    val = pd.read_csv(args.val)
    test= pd.read_csv(args.test)
    pp, py, gp, gy = pick_cols(val)

    Xv = val[[pp,py]].to_numpy(dtype=float)
    Yv = val[[gp,gy]].to_numpy(dtype=float)
    Xt = test[[pp,py]].to_numpy(dtype=float)

    # init
    I, H, O = 2, args.hidden, 2
    W1 = rs.randn(I,H)/np.sqrt(I)
    b1 = np.zeros(H)
    W2 = rs.randn(H,O)/np.sqrt(H)
    b2 = np.zeros(O)
    mW1=mW2=mb1=mb2=0.0

    def forward(X):
        Z1 = X@W1 + b1
        A1 = relu(Z1)
        Yh = A1@W2 + b2
        cache=(X,Z1,A1)
        return Yh, cache

    def step(X,Y,lr=0.01, beta=0.9, wd=1e-6):
        nonlocal W1,b1,W2,b2,mW1,mW2,mb1,mb2
        Yh,(X,Z1,A1)=forward(X)
        E = Yh - Y
        loss = np.mean(np.abs(E))
        # smooth L1-ish gradient via sign
        G = np.sign(E) / X.shape[0]   # [N,2]
        # backprop
        gW2 = A1.T @ G + wd*W2
        gb2 = G.sum(axis=0)
        dA1 = G @ W2.T
        dZ1 = dA1 * (Z1>0)
        gW1 = X.T @ dZ1 + wd*W1
        gb1 = dZ1.sum(axis=0)
        # momentum
        mW1 = beta*mW1 + (1-beta)*gW1
        mW2 = beta*mW2 + (1-beta)*gW2
        mb1 = beta*mb1 + (1-beta)*gb1
        mb2 = beta*mb2 + (1-beta)*gb2
        # update
        W1 -= lr*mW1; b1 -= lr*mb1
        W2 -= lr*mW2; b2 -= lr*mb2
        return float(loss)

    bestW = (W1.copy(),b1.copy(),W2.copy(),b2.copy()); best=1e9; patience=80; bad=0
    for e in range(args.epochs):
        L = step(Xv,Yv,lr=args.lr)
        if L < best-1e-5:
            best=L; bad=0
            bestW = (W1.copy(),b1.copy(),W2.copy(),b2.copy())
        else:
            bad+=1
            if bad>=patience: break

    W1,b1,W2,b2 = bestW
    # train MAE
    Yh,_ = forward(Xv)
    mae = np.mean(np.abs(Yh - Yv)); to_deg=180/np.pi
    print(f"[VAL mlp] MAE(rad)={mae:.4f} | deg={mae*to_deg:.2f}")

    # apply to test
    Yt,_ = forward(Xt)
    test["pitch_corr"] = Yt[:,0]; test["yaw_corr"] = Yt[:,1]
    test.to_csv(args.out, index=False); print(f"[WRITE] {args.out}")

if __name__=="__main__":
    main()


