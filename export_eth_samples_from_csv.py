#!/usr/bin/env python3
import os, argparse, re, numpy as np, pandas as pd, h5py
from PIL import Image

FRAME_RE = re.compile(r"^frame(\d+)$")

def to_img(arr):
    if arr.ndim == 2:
        from PIL import Image
        return Image.fromarray(arr).convert("RGB")
    if arr.ndim == 3 and arr.shape[-1] == 3:
        from PIL import Image
        return Image.fromarray(arr)
    # channel-first -> channel-last fallback
    from PIL import Image
    return Image.fromarray(np.moveaxis(arr, 0, -1))

def read_frame(f: h5py.File, ds_name: str, frame_key: str):
    """Return numpy array for the frame, supporting both Dataset and Group layouts."""
    if ds_name not in f:
        raise KeyError(f"'{ds_name}' not in H5")
    obj = f[ds_name]
    m = FRAME_RE.match(frame_key)
    if isinstance(obj, h5py.Dataset):
        # object is a single array of shape (N, H, W, C?) or (N, H, W)
        if not m:
            raise KeyError(f"Expected 'frameNNN' for Dataset indexing, got '{frame_key}'")
        idx = int(m.group(1))
        return obj[idx]  # h5py gives a numpy array slice
    elif isinstance(obj, h5py.Group):
        # object contains per-frame datasets like 'frame0', 'frame1', ...
        if frame_key not in obj:
            raise KeyError(f"'{frame_key}' not found in Group '{ds_name}'")
        return obj[frame_key][()]
    else:
        raise TypeError(f"Unsupported H5 object type for '{ds_name}': {type(obj)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/users/project1/pt01281/gaze_outputs/eth_eval_calib/preds_test_CALIB_piecewise.csv")
    ap.add_argument("--out", default="/users/kdm/divjots2002/gazemain/my_faces/eth_samples")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Read more than needed so we can skip any unreadables
    df = pd.read_csv(args.csv, usecols=["path"], nrows=max(args.n*20, args.n))
    saved, cache = 0, {}

    for p in df["path"]:
        p = str(p)
        if "::" not in p:
            continue
        h5_path, ds_name, frame_key = p.split("::")
        try:
            if h5_path not in cache:
                cache[h5_path] = h5py.File(h5_path, "r")
            f = cache[h5_path]
            arr = read_frame(f, ds_name, frame_key)
            img = to_img(arr)
            base = os.path.basename(h5_path).replace(".h5", "")
            outp = os.path.join(args.out, f"{base}_{frame_key}.png")
            img.save(outp)
            saved += 1
            if saved >= args.n:
                break
        except Exception as e:
            # Uncomment for debugging:
            # print("[SKIP]", h5_path, ds_name, frame_key, "|", e)
            continue

    for fh in cache.values():
        try: fh.close()
        except: pass

    print(f"[DONE] Saved {saved} samples to {args.out}")

if __name__ == "__main__":
    main()

