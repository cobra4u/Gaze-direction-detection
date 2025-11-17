#!/usr/bin/env python3
import argparse, os, h5py, numpy as np, pandas as pd, torch
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485,0.456,0.406]
IMAGENET_STD  = [0.229,0.224,0.225]

def build_model():
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    return m

def open_h5_cached(h5_path, cache):
    f = cache.get(h5_path)
    if f is None or not getattr(f, 'id', None):
        cache[h5_path] = h5py.File(h5_path, 'r')
    return cache[h5_path]

def load_frame_from_h5(f, ds_name, frame_key):
    """
    ds_name like 'face_patch'
    frame_key like 'frame123' -> index 123
    Supports arrays shaped:
      (N,H,W,3), (N,3,H,W), (N,H,W) grayscale
    Returns a PIL RGB image.
    """
    if ds_name not in f:
        raise KeyError(f"dataset '{ds_name}' not in H5")

    ds = f[ds_name]
    # parse 'frameNNN' -> int
    if frame_key.lower().startswith("frame"):
        idx = int(frame_key[5:])
    else:
        # fallback: if frame_key is already int-like
        try:
            idx = int(frame_key)
        except Exception:
            raise KeyError(f"bad frame key '{frame_key}'")

    arr = ds[idx]  # integer indexing into dataset

    # normalize to H,W,3 uint8 for PIL
    if arr.ndim == 2:  # (H,W) grayscale
        img = Image.fromarray(arr).convert("RGB")
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:     # (H,W,3)
            img = Image.fromarray(arr)
        elif arr.shape[0] == 3:    # (3,H,W)
            img = Image.fromarray(np.moveaxis(arr, 0, -1))
        else:
            # Unknown layout — try to coerce last dim to 3
            a = arr
            if a.shape[-1] != 3 and a.shape[0] != 3:
                raise ValueError(f"unexpected array shape {a.shape}")
            img = Image.fromarray(a if a.shape[-1]==3 else np.moveaxis(a,0,-1))
    else:
        raise ValueError(f"unexpected array ndim={arr.ndim}")
    return img.convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta-hflip", action="store_true", help="do horizontal flip TTA and median-combine")
    args = ap.parse_args()

    # collect rows from all parts
    part_files = [os.path.join(args.parts_dir, f) for f in os.listdir(args.parts_dir)
                  if f.endswith(".csv")]
    part_files.sort()
    frames = []
    for pf in part_files:
        df = pd.read_csv(pf)
        # Use only cols needed to re-extract H5 + GT
        need = ["subject_id","path","gt_pitch","gt_yaw"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"{pf} missing columns {miss}")
        frames.append(df[need])
        break  # one shard is enough (you can remove this break to use all)
    df = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] rows={len(df)} from {len(frames)} parts; cols={list(df.columns)}")

    # build + load model
    model = build_model().to(args.device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] Missing={len(missing)} Unexpected={len(unexpected)}")

    tfm = transforms.Compose([
        transforms.Resize((args.resize, args.resize), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    h5_cache = {}
    out_rows = []
    paths = df["path"].tolist()
    subs  = df["subject_id"].tolist()
    gtp   = df["gt_pitch"].tolist()
    gty   = df["gt_yaw"].tolist()

    def predict_batch(imgs):
        x = torch.stack(imgs, 0).to(args.device, non_blocking=True)
        with torch.no_grad():
            y = model(x).detach().cpu().numpy()
        return y

    total = len(paths)
    fails = 0
    batch_imgs, batch_idx = [], []

    for i in range(0, total, args.stride):
        p = paths[i]
        if "::" not in p:
            fails += 1
            continue
        h5_path, ds_name, frame_key = p.split("::", 2)
        try:
            f = open_h5_cached(h5_path, h5_cache)
            img = load_frame_from_h5(f, ds_name, frame_key)
            img_t = tfm(img)
        except Exception as e:
            # You can uncomment for debugging:
            # print(f"[WARN] failed {p} | {e}")
            fails += 1
            continue

        batch_imgs.append(img_t)
        batch_idx.append(i)
        if len(batch_imgs) == args.batch:
            y = predict_batch(batch_imgs)
            for k, (pp, yy) in enumerate(zip(batch_idx, y)):
                out_rows.append([subs[pp], paths[pp], gtp[pp], gty[pp], float(yy[0]), float(yy[1])])
            batch_imgs, batch_idx = [], []

    if batch_imgs:
        y = predict_batch(batch_imgs)
        for k, (pp, yy) in enumerate(zip(batch_idx, y)):
            out_rows.append([subs[pp], paths[pp], gtp[pp], gty[pp], float(yy[0]), float(yy[1])])

    out = pd.DataFrame(out_rows, columns=["subject_id","path","gt_pitch","gt_yaw","pr_pitch","pr_yaw"])
    out.to_csv(args.outcsv, index=False)
    print(f"[WRITE] {args.outcsv} rows: {len(out)} | fails: {fails} of {total}")

if __name__ == "__main__":
    main()

