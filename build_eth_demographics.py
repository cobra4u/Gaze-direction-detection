import os, argparse, glob, h5py, numpy as np, pandas as pd
from insightface.app import FaceAnalysis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_root", default="/users/project1/pt01281/dataset/xgaze_224/test")
    ap.add_argument("--out_csv", default="/users/project1/pt01281/gaze_outputs/eth_eval/tables/ETH_subject_demographics.csv")
    ap.add_argument("--sample_every", type=int, default=1000, help="sample every K frames per subject")
    ap.add_argument("--max_frames", type=int, default=10, help="cap frames sampled per subject")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    print("[LOAD] InsightFace (buffalo_l) on", args.device)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640,640))

    rows = []
    h5_paths = sorted(glob.glob(os.path.join(args.h5_root, "subject*.h5")))
    print(f"[SCAN] found {len(h5_paths)} subjects")
    for p in h5_paths:
        sid = os.path.splitext(os.path.basename(p))[0]  # e.g., subject0001
        try:
            with h5py.File(p, "r") as f:
                ds = f["face_patch"]
                n = ds.shape[0]
                idxs = list(range(0, n, args.sample_every))[:args.max_frames]
                genders, ages = [], []
                for i in idxs:
                    img = ds[i]            # (H,W,3) uint8
                    if img.ndim == 2:      # grayscale -> RGB
                        img = np.stack([img]*3, axis=-1)
                    bgr = img[..., ::-1]   # RGB->BGR for insightface
                    dets = app.get(bgr)
                    if len(dets) == 0: 
                        continue
                    d0 = dets[0]
                    # FaceAnalysis attaches sex/age if genderage.onnx is available
                    gender = getattr(d0, "gender", None)
                    age    = getattr(d0, "age", None)
                    if gender is None or age is None:
                        continue
                    genders.append(int(gender))  # 0=female, 1=male
                    ages.append(float(age))
                if len(genders) == 0:
                    print(f"[WARN] no demographic for {sid} (detections failed) — skipped")
                    continue
                # aggregate per subject
                g_majority = 1 if np.mean(genders) >= 0.5 else 0
                age_med = float(np.median(ages))
                if   age_med < 35: ab = "18-34"
                elif age_med < 55: ab = "35-54"
                else:              ab = "55+"
                rows.append([sid, "male" if g_majority==1 else "female", ab, len(genders)])
        except Exception as e:
            print(f"[WARN] {sid}: {e} — skipped")

    out = pd.DataFrame(rows, columns=["subject_id","gender","age_bin","N_frames_used"])
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}  subjects={len(out)}")

if __name__ == "__main__":
    main()
