import os, argparse, glob, h5py, numpy as np, pandas as pd

# Prefer cv2 for resizing; fallback to PIL if unavailable
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    from PIL import Image
    HAVE_CV2 = False

# Optional: InsightFace detection (best-effort)
HAVE_INSIGHT = True
try:
    from insightface.app import FaceAnalysis
except Exception:
    HAVE_INSIGHT = False

# Optional: ONNXRuntime for genderage fallback
HAVE_ORT = True
try:
    import onnxruntime as ort
except Exception:
    HAVE_ORT = False

def bgr_resize(img_bgr, size=(96,96)):
    if HAVE_CV2:
        return cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    else:
        # PIL expects RGB
        rgb = img_bgr[..., ::-1]
        im = Image.fromarray(rgb)
        im = im.resize(size, resample=Image.BILINEAR)
        out_rgb = np.asarray(im)
        return out_rgb[..., ::-1]

def run_genderage_onnx(img_bgr, sess, input_name=None):
    """
    img_bgr: uint8 HxWx3 (face crop)
    sess: onnxruntime.InferenceSession for genderage.onnx
    returns: (gender_int, age_float) if successful, else (None, None)
    """
    try:
        crop = bgr_resize(img_bgr, (96,96))
        x = crop.astype(np.float32) / 255.0  # 0..1
        x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
        if input_name is None:
            input_name = sess.get_inputs()[0].name
        outs = sess.run(None, {input_name: x})
        # Heuristic decode:
        # Most genderage heads return [gender_logits, age] or [age, gender_logits]
        g, a = None, None
        # try to find a 2-logit output
        gl = None
        af = None
        for out in outs:
            arr = np.asarray(out).squeeze()
            if arr.ndim == 1 and arr.size == 2:
                gl = arr
            elif arr.ndim == 0 or (arr.ndim==1 and arr.size==1):
                af = float(arr.reshape(-1)[0])
            elif arr.ndim==1 and arr.size>2 and arr.size<10:
                # sometimes genderage outputs age distribution; take mean index
                # but usually not in buffalo_l; ignore
                pass
        if gl is not None:
            g = int(np.argmax(gl))
        if af is not None:
            a = af
        return g, a
    except Exception:
        return None, None

def pad_square(img_bgr, pad=24):
    h, w, _ = img_bgr.shape
    top = bottom = left = right = pad
    return np.pad(img_bgr, ((top,bottom),(left,right),(0,0)), mode='reflect')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_root", default="/users/project1/pt01281/dataset/xgaze_224/test")
    ap.add_argument("--out_csv", default="/users/project1/pt01281/gaze_outputs/eth_eval/tables/ETH_subject_demographics.csv")
    ap.add_argument("--sample_every", type=int, default=1000, help="sample every K frames per subject")
    ap.add_argument("--max_frames", type=int, default=8, help="cap frames sampled per subject")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--try_pad", type=int, default=24, help="border padding for detection")
    ap.add_argument("--det_size", type=int, default=320, help="detector input size (square)")
    ap.add_argument("--genderage_onnx", default="/users/kdm/divjots2002/.insightface/models/buffalo_l/genderage.onnx")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    app = None
    if HAVE_INSIGHT:
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
            print(f"[INFO] InsightFace ready: det_size=({args.det_size},{args.det_size})")
        except Exception as e:
            print("[WARN] InsightFace init failed — will rely on ONNX fallback:", e)
            app = None
    else:
        print("[WARN] InsightFace not installed — will rely on ONNX fallback")

    sess = None
    if HAVE_ORT and os.path.exists(args.genderage_onnx):
        try:
            sess = ort.InferenceSession(args.genderage_onnx, providers=["CPUExecutionProvider"])
            print("[INFO] ONNX genderage session ready")
        except Exception as e:
            print("[WARN] genderage ONNX session failed:", e)
            sess = None
    else:
        print("[WARN] ONNXRuntime or genderage.onnx not available — fallback disabled")

    h5_paths = sorted(glob.glob(os.path.join(args.h5_root, "subject*.h5")))
    print(f"[SCAN] found {len(h5_paths)} subjects")

    rows = []
    for p in h5_paths:
        sid = os.path.splitext(os.path.basename(p))[0]
        genders, ages, used = [], [], 0
        try:
            with h5py.File(p, "r") as f:
                ds = f["face_patch"]
                n = ds.shape[0]
                idxs = list(range(0, n, args.sample_every))[:args.max_frames]
                for i in idxs:
                    img = ds[i]
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    if img.dtype != np.uint8:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    # BGR for insightface/onnx
                    bgr = img[..., ::-1]
                    g, a = None, None

                    # Try detector first (with padding context)
                    if app is not None:
                        test_bgr = pad_square(bgr, pad=args.try_pad) if args.try_pad>0 else bgr
                        dets = app.get(test_bgr)
                        if len(dets)>0:
                            d0 = dets[0]
                            g = getattr(d0, "gender", None)
                            a = getattr(d0, "age", None)

                    # Fallback: direct ONNX
                    if (g is None or a is None) and sess is not None:
                        g2, a2 = run_genderage_onnx(bgr, sess)
                        if g2 is not None: g = g2
                        if a2 is not None: a = a2

                    if g is not None and a is not None:
                        genders.append(int(g))
                        ages.append(float(a))
                        used += 1

            if used == 0:
                print(f"[WARN] {sid}: no usable frames (detector+onnx failed) — skipped")
                continue

            # Aggregate per subject
            g_majority = 1 if (np.mean(genders) >= 0.5) else 0  # 1=male 0=female
            age_med = float(np.median(ages))
            if   age_med < 35: ab = "18-34"
            elif age_med < 55: ab = "35-54"
            else:              ab = "55+"
            rows.append([sid, "male" if g_majority==1 else "female", ab, used])
            print(f"[OK] {sid}: gender={'male' if g_majority==1 else 'female'}  age_med={age_med:.1f} -> {ab}  (frames={used})")

        except Exception as e:
            print(f"[WARN] {sid}: {e} — skipped")
            continue

    out = pd.DataFrame(rows, columns=["subject_id","gender","age_bin","N_frames_used"])
    out.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}  subjects={len(out)}")

if __name__ == "__main__":
    main()
