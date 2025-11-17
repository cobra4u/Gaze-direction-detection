import os, argparse, glob, h5py, numpy as np, pandas as pd, sys, csv

# Optional libs
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    from PIL import Image
    HAVE_CV2 = False

try:
    from insightface.app import FaceAnalysis
    HAVE_INSIGHT = True
except Exception:
    HAVE_INSIGHT = False

try:
    import onnxruntime as ort
    HAVE_ORT = True
except Exception:
    HAVE_ORT = False

def bgr_resize(img_bgr, size=(96,96)):
    if HAVE_CV2:
        return cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    # PIL path
    rgb = img_bgr[..., ::-1]
    im = Image.fromarray(rgb).resize(size, resample=Image.BILINEAR)
    out_rgb = np.asarray(im)
    return out_rgb[..., ::-1]

def pad_square(img_bgr, pad=16):
    return np.pad(img_bgr, ((pad,pad),(pad,pad),(0,0)), mode='reflect')

def run_genderage_onnx(img_bgr, sess, input_name=None):
    try:
        crop = bgr_resize(img_bgr, (96,96)).astype(np.float32)/255.0
        x = np.transpose(crop, (2,0,1))[None, ...]  # NCHW
        if input_name is None:
            input_name = sess.get_inputs()[0].name
        outs = sess.run(None, {input_name: x})
        g, a = None, None
        gl, af = None, None
        for out in outs:
            arr = np.asarray(out).squeeze()
            if arr.ndim==1 and arr.size==2:
                gl = arr
            elif arr.ndim==0 or (arr.ndim==1 and arr.size==1):
                af = float(arr.reshape(-1)[0])
        if gl is not None: g = int(np.argmax(gl))
        if af is not None: a = af
        return g, a
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_root", default="/users/project1/pt01281/dataset/xgaze_224/test")
    ap.add_argument("--out_csv", default="/users/project1/pt01281/gaze_outputs/eth_eval/tables/ETH_subject_demographics.csv")
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--max_frames", type=int, default=6)
    ap.add_argument("--det_size", type=int, default=256)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--detect", choices=["auto","off","on"], default="auto",
                    help="auto: try detector then fallback; off: ONNX-only; on: detector-only")
    ap.add_argument("--genderage_onnx", default="/users/kdm/divjots2002/.insightface/models/buffalo_l/genderage.onnx")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Resume: load already done subjects
    done = set()
    if os.path.exists(args.out_csv):
        try:
            df_prev = pd.read_csv(args.out_csv)
            if "subject_id" in df_prev.columns:
                done = set(df_prev["subject_id"].astype(str).tolist())
                print(f"[RESUME] found {len(done)} subjects already in {args.out_csv}")
        except Exception as e:
            print("[WARN] could not read existing CSV, will recreate:", e)

    # Prepare detector (optional)
    app = None
    if args.detect in ("auto","on") and HAVE_INSIGHT:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
            print(f"[INFO] InsightFace ready: det_size=({args.det_size},{args.det_size})")
        except Exception as e:
            print("[WARN] InsightFace init failed:", e)
            if args.detect == "on":
                print("[FATAL] detect=on but detector unavailable.")
                sys.exit(1)
            app = None

    # Prepare ONNX genderage (optional)
    sess = None
    if args.detect in ("auto","off") and HAVE_ORT and os.path.exists(args.genderage_onnx):
        try:
            sess = ort.InferenceSession(args.genderage_onnx, providers=["CPUExecutionProvider"])
            print("[INFO] ONNX genderage session ready")
        except Exception as e:
            print("[WARN] genderage ONNX session failed:", e)
            if args.detect == "off":
                print("[FATAL] detect=off but genderage ONNX unavailable.")
                sys.exit(1)
            sess = None

    h5_paths = sorted(glob.glob(os.path.join(args.h5_root, "subject*.h5")))
    print(f"[SCAN] found {len(h5_paths)} subjects")

    # Open CSV in append or write mode, ensure header once
    need_header = not os.path.exists(args.out_csv) or (len(done)==0)
    f = open(args.out_csv, "a", newline="")
    w = csv.writer(f)
    if need_header:
        w.writerow(["subject_id","gender","age_bin","N_frames_used"])
        f.flush()

    try:
        for p in h5_paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            if sid in done:
                print(f"[SKIP] {sid} (already written)")
                continue

            genders, ages, used = [], [], 0
            try:
                with h5py.File(p, "r") as hf:
                    ds = hf["face_patch"]
                    n = ds.shape[0]
                    idxs = list(range(0, n, args.sample_every))[:args.max_frames]
                    for i in idxs:
                        img = ds[i]
                        if img.ndim == 2:
                            img = np.stack([img]*3, axis=-1)
                        if img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                        bgr = img[..., ::-1]

                        g, a = None, None
                        # detector path?
                        if app is not None and args.detect in ("auto","on"):
                            padded = pad_square(bgr, pad=args.pad) if args.pad>0 else bgr
                            dets = app.get(padded)
                            if len(dets)>0:
                                d0 = dets[0]
                                g = getattr(d0, "gender", None)
                                a = getattr(d0, "age", None)

                        # fallback ONNX?
                        if (g is None or a is None) and sess is not None and args.detect in ("auto","off"):
                            g2, a2 = run_genderage_onnx(bgr, sess)
                            if g2 is not None: g = g2
                            if a2 is not None: a = a2

                        if g is not None and a is not None:
                            genders.append(int(g))
                            ages.append(float(a))
                            used += 1

                if used == 0:
                    print(f"[WARN] {sid}: no usable frames — writing placeholder and continuing")
                    w.writerow([sid, "unknown", "unknown", 0]); f.flush()
                    continue

                g_majority = 1 if (np.mean(genders) >= 0.5) else 0  # 1=male, 0=female
                age_med = float(np.median(ages))
                if   age_med < 35: ab = "18-34"
                elif age_med < 55: ab = "35-54"
                else:              ab = "55+"
                w.writerow([sid, "male" if g_majority==1 else "female", ab, used]); f.flush()
                print(f"[OK] {sid}: gender={'male' if g_majority==1 else 'female'} age_med={age_med:.1f} -> {ab} (frames={used})")

            except Exception as e:
                print(f"[WARN] {sid}: {e} — writing placeholder and continuing")
                w.writerow([sid, "unknown", "unknown", 0]); f.flush()

    finally:
        f.close()
        print(f"[DONE] wrote (or appended) -> {args.out_csv}")

if __name__ == "__main__":
    main()
