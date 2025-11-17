#!/usr/bin/env python3
import os, glob, random, h5py
from PIL import Image
import numpy as np

def save_png(arr, out_path):
    if arr.ndim == 2:
        img = Image.fromarray(arr).convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        img = Image.fromarray(arr)
    else:
        # channel-first -> channel-last
        img = Image.fromarray(np.moveaxis(arr, 0, -1))
    img.save(out_path)

def main():
    src_root = "/users/project1/pt01281/dataset/xgaze_224/test"
    dst_root = "/users/kdm/divjots2002/gazemain/my_faces/eth_samples"
    os.makedirs(dst_root, exist_ok=True)

    h5_files = sorted(glob.glob(os.path.join(src_root, "subject*.h5")))
    random.seed(42)
    random.shuffle(h5_files)

    saved = 0
    target = 15  # save ~15 images
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                if "face_patch" not in f: 
                    continue
                ds = f["face_patch"]
                # random 2–4 frames from this subject
                idxs = list(range(len(ds)))
                random.shuffle(idxs)
                for i in idxs[:3]:
                    try:
                        arr = ds[f"frame{i}"]
                        name = os.path.splitext(os.path.basename(h5_path))[0]
                        out = os.path.join(dst_root, f"{name}_frame{i}.png")
                        save_png(arr, out)
                        saved += 1
                        if saved >= target:
                            print(f"[DONE] Saved {saved} samples to {dst_root}")
                            return
                    except Exception:
                        continue
        except Exception:
            continue
    print(f"[DONE] Saved {saved} samples to {dst_root}")

if __name__ == "__main__":
    main()

